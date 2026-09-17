import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import topotoolbox as tt3
from scipy import ndimage

from model import create_model


TILE_SIZE = 64
STRIDE = 32
NUM_CLASSES = 5

# Larger batches keep modern GPUs busier than the previous value of 256.
# You can override this without editing the script by setting:
# HABITAT_BATCH_SIZE=512, 1024, 2048, etc.
DEFAULT_CUDA_BATCH_SIZE = 1024
DEFAULT_CPU_BATCH_SIZE = 64

# Model class index for Bare Terrain (GIS class 5 after +1 conversion)
BARE_CLASS = 4


def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    return data, profile


def make_positions(length, tile_size, stride):
    positions = list(
        range(
            0,
            length - tile_size + 1,
            stride,
        )
    )

    last = length - tile_size

    if positions[-1] != last:
        positions.append(last)

    return positions


def calculate_normalized_flow(dem_path):
    """
    Reproduce the flow preprocessing:
    raw flow accumulation -> log1p -> min/max normalize to 0-1.
    """

    print("Calculating flow accumulation for Bare Terrain refinement...")

    dem = tt3.read_tif(str(dem_path))
    fd = tt3.FlowObject(dem)
    acc = fd.flow_accumulation()

    raw_flow = np.asarray(
        acc.z,
        dtype=np.float32,
    )

    flow_log = np.log1p(
        np.maximum(raw_flow, 0.0)
    )

    valid = np.isfinite(flow_log)

    if not np.any(valid):
        raise RuntimeError(
            "Flow accumulation contains no valid values."
        )

    flow_min = float(
        np.nanmin(flow_log[valid])
    )
    flow_max = float(
        np.nanmax(flow_log[valid])
    )

    if flow_max <= flow_min:
        raise RuntimeError(
            "Flow accumulation has no usable value range."
        )

    flow_accumulation = np.zeros_like(
        flow_log,
        dtype=np.float32,
    )

    flow_accumulation[valid] = (
        flow_log[valid] - flow_min
    ) / (
        flow_max - flow_min
    )

    flow_accumulation = np.clip(
        flow_accumulation,
        0.0,
        1.0,
    )

    print(
        "Normalized flow range:",
        float(np.nanmin(flow_accumulation)),
        "to",
        float(np.nanmax(flow_accumulation)),
    )

    return flow_accumulation


def remove_isolated_bare_patches(
    prediction,
    bare_class=5,
    min_pixels=20,
    proximity_pixels=20,
    min_nearby_components=8,
):
    """
    Faster equivalent of the original Bare Terrain cleanup.

    Small components are processed only inside a local bounding box,
    instead of repeatedly dilating a full-raster mask.
    """

    cleaned = prediction.copy()
    bare_mask = cleaned == bare_class

    labelled, num_features = ndimage.label(
        bare_mask
    )

    if num_features == 0:
        return cleaned

    sizes = np.bincount(
        labelled.ravel()
    )

    objects = ndimage.find_objects(
        labelled
    )

    removed_components = 0
    removed_pixels = 0

    height, width = prediction.shape

    for component_id in range(
        1,
        num_features + 1,
    ):
        size = sizes[component_id]

        if size >= min_pixels:
            continue

        obj = objects[
            component_id - 1
        ]

        if obj is None:
            continue

        y_slice, x_slice = obj

        y0 = max(
            0,
            y_slice.start - proximity_pixels,
        )
        y1 = min(
            height,
            y_slice.stop + proximity_pixels,
        )
        x0 = max(
            0,
            x_slice.start - proximity_pixels,
        )
        x1 = min(
            width,
            x_slice.stop + proximity_pixels,
        )

        local_labels = labelled[
            y0:y1,
            x0:x1,
        ]

        component_local = (
            local_labels == component_id
        )

        expanded_local = ndimage.binary_dilation(
            component_local,
            iterations=proximity_pixels,
        )

        nearby_labels = np.unique(
            local_labels[
                expanded_local
                & ~component_local
            ]
        )

        nearby_labels = nearby_labels[
            (nearby_labels != 0)
            & (
                nearby_labels
                != component_id
            )
        ]

        if (
            len(nearby_labels)
            >= min_nearby_components
        ):
            continue

        neighbour_ring_local = (
            ndimage.binary_dilation(
                component_local,
                iterations=1,
            )
            & ~component_local
        )

        local_cleaned = cleaned[
            y0:y1,
            x0:x1,
        ]

        neighbours = local_cleaned[
            neighbour_ring_local
        ]

        neighbours = neighbours[
            (neighbours != 255)
            & (
                neighbours
                != bare_class
            )
        ]

        if neighbours.size > 0:
            replacement = np.bincount(
                neighbours.astype(
                    np.int64
                )
            ).argmax()

            local_cleaned[
                component_local
            ] = replacement

            removed_components += 1
            removed_pixels += int(size)

    print(
        "Bare tidy-up removed components:",
        removed_components,
    )
    print(
        "Bare tidy-up replaced pixels:",
        removed_pixels,
    )

    return cleaned


def smooth_prediction(
    prediction,
    device,
    size=3,
):
    """
    3x3 majority smoothing for habitat classes 1-5.

    Uses CUDA when available, otherwise falls back to SciPy.
    """

    if size != 3:
        raise ValueError(
            "This optimized smoother currently supports size=3 only."
        )

    if device.type == "cuda":

        pred = torch.from_numpy(
            prediction.astype(
                np.int64,
                copy=False,
            )
        ).to(
            device,
            non_blocking=True,
        )

        best_count = torch.full(
            pred.shape,
            -1.0,
            dtype=torch.float32,
            device=device,
        )

        best_class = torch.ones(
            pred.shape,
            dtype=torch.uint8,
            device=device,
        )

        kernel = torch.ones(
            (1, 1, 3, 3),
            dtype=torch.float32,
            device=device,
        )

        # Process one class at a time to keep VRAM use low.
        for class_value in range(
            1,
            NUM_CLASSES + 1,
        ):
            mask = (
                pred == class_value
            ).to(
                torch.float32
            )[None, None, :, :]

            counts = F.conv2d(
                mask,
                kernel,
                padding=1,
            )[0, 0]

            # Strictly greater preserves lower-class tie breaking.
            update = counts > best_count

            best_count[
                update
            ] = counts[
                update
            ]

            best_class[
                update
            ] = class_value

        best_class[
            pred == 255
        ] = 255

        result = (
            best_class
            .cpu()
            .numpy()
        )

        return result

    # CPU fallback
    class_counts = np.empty(
        (
            NUM_CLASSES,
            prediction.shape[0],
            prediction.shape[1],
        ),
        dtype=np.uint8,
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    for class_value in range(
        1,
        NUM_CLASSES + 1,
    ):
        class_mask = (
            prediction == class_value
        ).astype(np.uint8)

        class_counts[
            class_value - 1
        ] = ndimage.convolve(
            class_mask,
            kernel,
            mode="nearest",
        )

    smoothed = (
        np.argmax(
            class_counts,
            axis=0,
        ).astype(np.uint8)
        + 1
    )

    smoothed[
        prediction == 255
    ] = 255

    return smoothed


def run_prediction(
    b02_path,
    b03_path,
    b04_path,
    dem_path,
    model_path,
    output_path,
    enable_flow_refinement,
    flow_threshold,
    bare_min_probability,
    flow_boost,
):

    total_start = time.perf_counter()

    # ------------------------------------------------------------
    # VALIDATE SETTINGS
    # ------------------------------------------------------------

    if not 0.0 <= flow_threshold < 1.0:
        raise ValueError(
            "Normalised flow threshold must be >= 0 and < 1."
        )

    if not 0.0 <= bare_min_probability <= 1.0:
        raise ValueError(
            "Minimum Bare Terrain probability must be between 0 and 1."
        )

    if flow_boost < 0.0:
        raise ValueError(
            "Flow boost must be >= 0."
        )

    # ------------------------------------------------------------
    # DEVICE / CUDA PERFORMANCE SETTINGS
    # ------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

        # Allow Tensor Core friendly TF32 paths where PyTorch can use them.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(
            "high"
        )

    batch_size = int(
        os.environ.get(
            "HABITAT_BATCH_SIZE",
            (
                DEFAULT_CUDA_BATCH_SIZE
                if device.type == "cuda"
                else DEFAULT_CPU_BATCH_SIZE
            ),
        )
    )

    print("Using device:", device)
    print("Inference batch size:", batch_size)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        total_memory = (
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU memory: {total_memory:.1f} GB"
        )

    print(
        "Bare Terrain flow refinement:",
        "ENABLED"
        if enable_flow_refinement
        else "DISABLED",
    )

    if enable_flow_refinement:
        print(
            "Flow settings:",
            f"threshold={flow_threshold},",
            f"min_bare_probability={bare_min_probability},",
            f"boost={flow_boost}",
        )

    # ------------------------------------------------------------
    # LOAD RASTERS
    # ------------------------------------------------------------

    b02, profile = read_raster(
        b02_path
    )
    b03, _ = read_raster(
        b03_path
    )
    b04, _ = read_raster(
        b04_path
    )

    if not (
        b02.shape
        == b03.shape
        == b04.shape
    ):
        raise ValueError(
            "B02, B03 and B04 do not have matching dimensions."
        )

    # ------------------------------------------------------------
    # SCALE SENTINEL
    # ------------------------------------------------------------
    b02 = np.clip(
        b02 / 10000.0,
        0.0,
        1.0,
    )
    b03 = np.clip(
        b03 / 10000.0,
        0.0,
        1.0,
    )
    b04 = np.clip(
        b04 / 10000.0,
        0.0,
        1.0,
    )

    height, width = b04.shape

    if (
        height < TILE_SIZE
        or width < TILE_SIZE
    ):
        raise ValueError(
            "Selected extent is smaller than the 64x64 model tile size."
        )

    # ------------------------------------------------------------
    # FLOW ACCUMULATION
    # ------------------------------------------------------------

    flow_accumulation = None

    if enable_flow_refinement:
        flow_start = time.perf_counter()

        flow_accumulation = (
            calculate_normalized_flow(
                dem_path
            )
        )

        print(
            f"Flow accumulation time: "
            f"{time.perf_counter() - flow_start:.2f} s"
        )

        if flow_accumulation.shape != b04.shape:
            raise ValueError(
                "Flow accumulation and Sentinel rasters do not have "
                "matching dimensions: "
                f"{flow_accumulation.shape} vs {b04.shape}."
            )

    # ------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------

    model_start = time.perf_counter()

    model = create_model(
        input_channels=3,
        number_of_classes=NUM_CLASSES,
    )

    state_dict = torch.load(
        model_path,
        map_location="cpu",
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        device
    )

    if device.type == "cuda":
        # channels_last is often faster for convolution-heavy models.
        model = model.to(
            memory_format=torch.channels_last
        )

    model.eval()

    print(
        f"Model load time: "
        f"{time.perf_counter() - model_start:.2f} s"
    )

    # ------------------------------------------------------------
    # SLIDING WINDOW POSITIONS
    # ------------------------------------------------------------

    x_positions = make_positions(
        width,
        TILE_SIZE,
        STRIDE,
    )
    y_positions = make_positions(
        height,
        TILE_SIZE,
        STRIDE,
    )

    total_tiles = (
        len(x_positions)
        * len(y_positions)
    )

    print(
        "Total inference tiles:",
        total_tiles,
    )

    # ------------------------------------------------------------
    # HANN WEIGHTING
    # ------------------------------------------------------------

    one_d = np.hanning(
        TILE_SIZE
    )

    weight_window = np.outer(
        one_d,
        one_d,
    ).astype(np.float32)

    weight_window = np.maximum(
        weight_window,
        1e-3,
    )

    # ------------------------------------------------------------
    # CHOOSE GPU OR CPU ACCUMULATION
    # ------------------------------------------------------------

    use_gpu_accumulation = False

    if device.type == "cuda":
        free_memory, _ = (
            torch.cuda.mem_get_info()
        )

        accumulator_bytes = (
            (NUM_CLASSES + 1)
            * height
            * width
            * 4
        )

        # Only reserve full prediction accumulators on the GPU if doing
        # so uses less than ~35% of currently free VRAM. This leaves room
        # for the model and the large inference batch.
        if (
            accumulator_bytes
            < free_memory * 0.35
        ):
            use_gpu_accumulation = True

    print(
        "Prediction accumulation:",
        (
            "GPU"
            if use_gpu_accumulation
            else "CPU"
        ),
    )

    if use_gpu_accumulation:

        score_sum_gpu = torch.zeros(
            (
                NUM_CLASSES,
                height * width,
            ),
            dtype=torch.float32,
            device=device,
        )

        weight_sum_gpu = torch.zeros(
            height * width,
            dtype=torch.float32,
            device=device,
        )

        weight_window_gpu = torch.from_numpy(
            weight_window
        ).to(
            device=device,
            dtype=torch.float32,
        )

        weight_flat_gpu = (
            weight_window_gpu
            .reshape(-1)
        )

        # Flat offsets for one 64x64 tile in the output raster.
        yy = torch.arange(
            TILE_SIZE,
            device=device,
            dtype=torch.long,
        )
        xx = torch.arange(
            TILE_SIZE,
            device=device,
            dtype=torch.long,
        )

        local_offsets_gpu = (
            yy[:, None] * width
            + xx[None, :]
        ).reshape(-1)

    else:

        score_sum = np.zeros(
            (
                NUM_CLASSES,
                height,
                width,
            ),
            dtype=np.float32,
        )

        weight_sum = np.zeros(
            (height, width),
            dtype=np.float32,
        )

    # ------------------------------------------------------------
    # BATCHED INFERENCE
    # ------------------------------------------------------------

    inference_start = time.perf_counter()

    completed = 0
    batch_tiles = []
    batch_positions = []

    refinement_candidate_count = 0


    def process_batch(
        batch_tiles,
        batch_positions,
    ):
        """
        Run one batch through the network.

        Performance changes compared with the earlier runner:
        - larger batches
        - pinned CPU memory + non-blocking CUDA transfer
        - channels_last CNN layout
        - FP16 autocast
        - optional GPU-side Hann accumulation
        """

        nonlocal refinement_candidate_count

        if not batch_tiles:
            return

        batch_np = np.stack(
            batch_tiles,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        batch_cpu = torch.from_numpy(
            batch_np
        )

        if device.type == "cuda":
            batch_cpu = (
                batch_cpu.pin_memory()
            )

            batch = batch_cpu.to(
                device,
                non_blocking=True,
            )

            batch = batch.contiguous(
                memory_format=torch.channels_last
            )
        else:
            batch = batch_cpu.to(
                device
            )

        with torch.inference_mode():

            with torch.autocast(
                device_type=device.type,
                enabled=(
                    device.type == "cuda"
                ),
            ):

                logits = model(
                    batch
                )

                probabilities = (
                    torch.softmax(
                        logits,
                        dim=1,
                    )
                )

                # ------------------------------------------------------------
                # BARE TERRAIN FLOW REFINEMENT
                # ------------------------------------------------------------

                if enable_flow_refinement:

                    flow_tiles_np = np.stack(
                        [
                            flow_accumulation[
                                y:y + TILE_SIZE,
                                x:x + TILE_SIZE,
                            ]
                            for y, x
                            in batch_positions
                        ],
                        axis=0,
                    ).astype(
                        np.float32,
                        copy=False,
                    )

                    flow_cpu = (
                        torch.from_numpy(
                            flow_tiles_np
                        )
                    )

                    if device.type == "cuda":
                        flow_cpu = (
                            flow_cpu.pin_memory()
                        )

                    flow_tensor = flow_cpu.to(
                        device,
                        non_blocking=(
                            device.type
                            == "cuda"
                        ),
                    )

                    bare_probability = probabilities[
                        :,
                        BARE_CLASS,
                        :,
                        :
                    ]

                    candidate = (
                        (
                            bare_probability
                            > bare_min_probability
                        )
                        & (
                            flow_tensor
                            > flow_threshold
                        )
                    )

                    flow_strength = (
                        flow_tensor
                        - flow_threshold
                    ) / (
                        1.0
                        - flow_threshold
                    )

                    flow_strength = (
                        torch.clamp(
                            flow_strength,
                            0.0,
                            1.0,
                        )
                    )

                    bare_probability = (
                        bare_probability
                        + candidate.to(
                            bare_probability.dtype
                        )
                        * flow_boost
                        * flow_strength
                    )

                    probabilities[
                        :,
                        BARE_CLASS,
                        :,
                        :
                    ] = bare_probability

                    probabilities = (
                        probabilities
                        / probabilities.sum(
                            dim=1,
                            keepdim=True,
                        )
                    )

                    refinement_candidate_count += int(
                        candidate.sum().item()
                    )

        # ------------------------------------------------------------
        # ACCUMULATE OVERLAPPING TILES
        # ------------------------------------------------------------

        if use_gpu_accumulation:

            bases_gpu = torch.tensor(
                [
                    y * width + x
                    for y, x
                    in batch_positions
                ],
                dtype=torch.long,
                device=device,
            )

            indices_gpu = (
                bases_gpu[:, None]
                + local_offsets_gpu[
                    None,
                    :
                ]
            )

            batch_count = (
                len(batch_positions)
            )

            weighted_probs = (
                probabilities
                * weight_window_gpu[
                    None,
                    None,
                    :,
                    :
                ]
            )

            weighted_probs = (
                weighted_probs
                .permute(
                    1,
                    0,
                    2,
                    3,
                )
                .reshape(
                    NUM_CLASSES,
                    -1,
                )
                .float()
            )

            expanded_indices = (
                indices_gpu
                .reshape(
                    1,
                    -1,
                )
                .expand(
                    NUM_CLASSES,
                    -1,
                )
            )

            score_sum_gpu.scatter_add_(
                1,
                expanded_indices,
                weighted_probs,
            )

            weight_values = (
                weight_flat_gpu[
                    None,
                    :
                ]
                .expand(
                    batch_count,
                    -1,
                )
                .reshape(-1)
            )

            weight_sum_gpu.scatter_add_(
                0,
                indices_gpu.reshape(
                    -1
                ),
                weight_values,
            )

        else:

            probabilities_np = (
                probabilities
                .float()
                .cpu()
                .numpy()
            )

            for probs, (y, x) in zip(
                probabilities_np,
                batch_positions,
            ):

                y2 = y + TILE_SIZE
                x2 = x + TILE_SIZE

                score_sum[
                    :,
                    y:y2,
                    x:x2,
                ] += (
                    probs
                    * weight_window[
                        np.newaxis,
                        :,
                        :
                    ]
                )

                weight_sum[
                    y:y2,
                    x:x2,
                ] += weight_window


    for y in y_positions:

        for x in x_positions:

            y2 = y + TILE_SIZE
            x2 = x + TILE_SIZE

            tile = np.stack(
                [
                    b04[y:y2, x:x2],
                    b03[y:y2, x:x2],
                    b02[y:y2, x:x2],
                ],
                axis=0,
            )

            batch_tiles.append(
                tile
            )
            batch_positions.append(
                (y, x)
            )

            if (
                len(batch_tiles)
                >= batch_size
            ):

                process_batch(
                    batch_tiles,
                    batch_positions,
                )

                completed += len(
                    batch_tiles
                )

                print(
                    f"Processed "
                    f"{completed}/"
                    f"{total_tiles} tiles"
                )

                batch_tiles = []
                batch_positions = []

    if batch_tiles:

        process_batch(
            batch_tiles,
            batch_positions,
        )

        completed += len(
            batch_tiles
        )

        print(
            f"Processed "
            f"{completed}/"
            f"{total_tiles} tiles"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(
        f"ML inference + tile accumulation time: "
        f"{time.perf_counter() - inference_start:.2f} s"
    )

    if enable_flow_refinement:
        print(
            "Bare Terrain refinement candidate tile-pixels:",
            refinement_candidate_count,
        )

    # ------------------------------------------------------------
    # COMBINE OVERLAPPING PREDICTIONS
    # ------------------------------------------------------------

    if use_gpu_accumulation:

        valid_gpu = (
            weight_sum_gpu > 0
        )

        score_sum_gpu[
            :,
            valid_gpu,
        ] /= (
            weight_sum_gpu[
                valid_gpu
            ]
        )

        prediction_gpu = (
            torch.argmax(
                score_sum_gpu,
                dim=0,
            )
            .to(
                torch.uint8
            )
            + 1
        )

        prediction_gpu[
            ~valid_gpu
        ] = 255

        prediction = (
            prediction_gpu
            .reshape(
                height,
                width,
            )
            .cpu()
            .numpy()
        )

        valid_prediction = (
            valid_gpu
            .reshape(
                height,
                width,
            )
            .cpu()
            .numpy()
        )

        # Release large GPU accumulators before CPU post-processing.
        del score_sum_gpu
        del weight_sum_gpu
        del prediction_gpu

        torch.cuda.empty_cache()

    else:

        valid_prediction = (
            weight_sum > 0
        )

        score_sum[
            :,
            valid_prediction,
        ] /= weight_sum[
            valid_prediction
        ]

        prediction = np.argmax(
            score_sum,
            axis=0,
        ).astype(np.uint8)

        prediction += 1

        prediction[
            ~valid_prediction
        ] = 255

    # ---------------------------------------------------------
    # POST-PROCESSING
    # ---------------------------------------------------------

    bare_pixels_before = int(
        np.count_nonzero(
            prediction == 5
        )
    )

    print(
        "Bare Terrain pixels before post-processing:",
        bare_pixels_before,
    )

    post_start = time.perf_counter()

    print(
        "Applying Bare Terrain tidy-up..."
    )

    prediction_clean = (
        remove_isolated_bare_patches(
            prediction,
            bare_class=5,
            min_pixels=20,
            proximity_pixels=20,
            min_nearby_components=8,
        )
    )

    print(
        "Applying 3x3 habitat-boundary smoothing..."
    )

    prediction_final = (
        smooth_prediction(
            prediction_clean,
            device=device,
            size=3,
        )
    )

    prediction_final[
        ~valid_prediction
    ] = 255

    bare_pixels_after = int(
        np.count_nonzero(
            prediction_final == 5
        )
    )

    changed_pixels = int(
        np.count_nonzero(
            (
                prediction_final
                != prediction
            )
            & valid_prediction
        )
    )

    print(
        "Bare Terrain pixels after post-processing:",
        bare_pixels_after,
    )

    print(
        "Total habitat pixels changed by post-processing:",
        changed_pixels,
    )

    print(
        f"Post-processing time: "
        f"{time.perf_counter() - post_start:.2f} s"
    )

    # ------------------------------------------------------------
    # SAVE OUTPUT
    # ------------------------------------------------------------

    profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=255,
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_start = time.perf_counter()

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:
        dst.write(
            prediction_final,
            1,
        )

    print(
        "Saved prediction:",
        output_path,
    )

    print(
        f"Raster save time: "
        f"{time.perf_counter() - save_start:.2f} s"
    )

    print(
        f"TOTAL inference_runner time: "
        f"{time.perf_counter() - total_start:.2f} s"
    )


if __name__ == "__main__":

    if len(sys.argv) != 11:
        raise SystemExit(
            "Usage:\n"
            "python inference_runner.py "
            "<B02> <B03> <B04> <DEM> <MODEL> <OUTPUT> "
            "<ENABLE_FLOW_REFINEMENT> <FLOW_THRESHOLD> "
            "<BARE_MIN_PROBABILITY> <FLOW_BOOST>"
        )

    b02_path = sys.argv[1]
    b03_path = sys.argv[2]
    b04_path = sys.argv[3]
    dem_path = sys.argv[4]
    model_path = sys.argv[5]
    output_path = sys.argv[6]

    enable_flow_refinement = (
        sys.argv[7]
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
    )

    flow_threshold = float(
        sys.argv[8]
    )

    bare_min_probability = float(
        sys.argv[9]
    )

    flow_boost = float(
        sys.argv[10]
    )

    run_prediction(
        b02_path,
        b03_path,
        b04_path,
        dem_path,
        model_path,
        output_path,
        enable_flow_refinement,
        flow_threshold,
        bare_min_probability,
        flow_boost,
    )
