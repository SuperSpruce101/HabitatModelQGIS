import sys
from pathlib import Path

import numpy as np
import rasterio
import topotoolbox as tt3


NODATA = 255


def create_drainage_network(
    dem_path,
    output_path,
    threshold,
):

    print("Loading DEM...")
    print(f"User river threshold: {threshold}")

    # ============================================================
    # FLOW ACCUMULATION
    # ============================================================

    dem = tt3.read_tif(dem_path)

    print("Calculating flow accumulation...")

    fd = tt3.FlowObject(dem)
    acc = fd.flow_accumulation()

    raw_flow = np.asarray(
        acc.z,
        dtype=np.float32,
    )

    print(
        "Raw flow range:",
        float(np.nanmin(raw_flow)),
        "to",
        float(np.nanmax(raw_flow)),
    )

    # ============================================================
    # SAME FLOW TRANSFORMATION USED BY ORIGINAL WORKFLOW
    # ============================================================

    # Compress the extremely skewed flow-accumulation values
    flow_log = np.log1p(
        np.maximum(raw_flow, 0)
    )

    valid = np.isfinite(flow_log)

    if not np.any(valid):
        raise RuntimeError(
            "No valid flow-accumulation values were produced."
        )

    flow_min = np.nanmin(
        flow_log[valid]
    )

    flow_max = np.nanmax(
        flow_log[valid]
    )

    if flow_max <= flow_min:
        raise RuntimeError(
            "Flow accumulation has no usable range."
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
        "Normalised flow range:",
        float(np.nanmin(flow_accumulation)),
        "to",
        float(np.nanmax(flow_accumulation)),
    )

    # ============================================================
    # RIVER MASK
    #
    # Equivalent to original:
    #
    # river_mask = (
    #     flow_accumulation >= RIVER_FLOW_THRESHOLD
    # )
    # ============================================================

    river_mask = (
        flow_accumulation >= threshold
    )

    print(
        f"Pixels >= {threshold}:",
        int(np.sum(river_mask)),
    )

    print(
        "Percentage of valid area classified as river:",
        round(
            np.sum(river_mask)
            / river_mask.size
            * 100,
            3,
        ),
        "%",
    )

    # ============================================================
    # OUTPUT RASTER
    #
    # 1   = river -> QGIS styles this red
    # 255 = NoData -> transparent
    # ============================================================

    drainage = np.full(
        flow_accumulation.shape,
        NODATA,
        dtype=np.uint8,
    )

    drainage[river_mask] = 1

    # ============================================================
    # SAVE
    # ============================================================

    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()

    profile.update(
        dtype="uint8",
        count=1,
        nodata=NODATA,
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            drainage,
            1,
        )

    print(
        "Saved river network:",
        output_path,
    )


if __name__ == "__main__":

    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: drainage_network.py "
            "<DEM> <OUTPUT> <RIVER_THRESHOLD>"
        )

    create_drainage_network(
        dem_path=sys.argv[1],
        output_path=sys.argv[2],
        threshold=float(sys.argv[3]),
    )