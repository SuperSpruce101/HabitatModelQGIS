import sys
from pathlib import Path

import numpy as np
import rasterio

from scipy.ndimage import uniform_filter


FOREST_CLASS = 1
NODATA = 255


def create_forest_subclasses(
    prediction_path,
    dem_path,
    output_path,
    slope_threshold,
    tpi_threshold,
    tpi_window,
):

    # ========================================================
    # LOAD HABITAT PREDICTION
    # ========================================================

    with rasterio.open(prediction_path) as src:
        prediction = src.read(1)
        profile = src.profile.copy()
        transform = src.transform

    # ========================================================
    # LOAD DEM
    # ========================================================

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        dem_nodata = src.nodata

    if prediction.shape != dem.shape:
        raise ValueError(
            "Prediction and aligned DEM do not have matching dimensions."
        )

    # ========================================================
    # VALID DEM MASK
    # ========================================================

    valid_dem = np.isfinite(dem)

    if dem_nodata is not None:
        valid_dem &= dem != dem_nodata

    # ========================================================
    # TPI
    # ========================================================

    valid_float = valid_dem.astype(np.float32)

    dem_for_filter = np.where(
        valid_dem,
        dem,
        0.0,
    )

    local_dem = uniform_filter(
        dem_for_filter,
        size=tpi_window,
        mode="nearest",
    )

    local_valid = uniform_filter(
        valid_float,
        size=tpi_window,
        mode="nearest",
    )

    local_mean = np.divide(
        local_dem,
        local_valid,
        out=np.full_like(
            dem,
            np.nan,
            dtype=np.float32,
        ),
        where=local_valid > 0,
    )

    tpi = dem - local_mean

    # ========================================================
    # SLOPE
    # ========================================================

    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)

    fallback = np.nanmedian(
        dem[valid_dem]
    )

    filled_dem = np.where(
        valid_dem,
        dem,
        fallback,
    )

    dz_dy, dz_dx = np.gradient(
        filled_dem,
        pixel_height,
        pixel_width,
    )

    slope = np.degrees(
        np.arctan(
            np.sqrt(
                dz_dx ** 2
                + dz_dy ** 2
            )
        )
    )

    # ========================================================
    # FOREST MASK
    # ========================================================

    forest = (
        (prediction == FOREST_CLASS)
        & valid_dem
    )

    # ========================================================
    # CLASSIFICATION
    #
    # 1 = Gentle / Plateau Forest
    # 2 = Steep-Slope Forest
    # 3 = Valley Forest
    # 255 = Not Forest / NoData
    # ========================================================

    output = np.full(
        prediction.shape,
        NODATA,
        dtype=np.uint8,
    )

    steep = (
        forest
        & (slope >= slope_threshold)
    )

    valley = (
        forest
        & (~steep)
        & (tpi <= tpi_threshold)
    )

    gentle = (
        forest
        & (~steep)
        & (~valley)
    )

    output[gentle] = 1
    output[steep] = 2
    output[valley] = 3

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    total_forest = np.sum(forest)

    print(
        f"Forest pixels: {total_forest}"
    )

    if total_forest > 0:

        print(
            "Gentle / Plateau Forest:",
            f"{100 * np.sum(gentle) / total_forest:.2f}%"
        )

        print(
            "Steep-Slope Forest:",
            f"{100 * np.sum(steep) / total_forest:.2f}%"
        )

        print(
            "Valley Forest:",
            f"{100 * np.sum(valley) / total_forest:.2f}%"
        )

    # ========================================================
    # SAVE
    # ========================================================

    profile.update(
        dtype=rasterio.uint8,
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
            output,
            1,
        )

    print(
        "Saved Forest subclasses:",
        output_path,
    )


if __name__ == "__main__":

    if len(sys.argv) != 7:

        raise SystemExit(
            "Usage:\n"
            "python forest_subclasses.py "
            "<PREDICTION> "
            "<DEM> "
            "<OUTPUT> "
            "<SLOPE_THRESHOLD> "
            "<TPI_THRESHOLD> "
            "<TPI_WINDOW>"
        )

    create_forest_subclasses(
        prediction_path=sys.argv[1],
        dem_path=sys.argv[2],
        output_path=sys.argv[3],
        slope_threshold=float(sys.argv[4]),
        tpi_threshold=float(sys.argv[5]),
        tpi_window=int(sys.argv[6]),
    )