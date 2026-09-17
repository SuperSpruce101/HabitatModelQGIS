import sys
from pathlib import Path

import numpy as np
import rasterio


L2A_DN_OFFSET = 1000
RT_MIN = 0
RT_MAX = 10000
RT_NODATA = 65535
RT_SCALE = 0.0001


def convert_l2a_to_rt(input_path, output_path):
    """
    Convert a Sentinel-2 L2A band to the RT-style integer representation
    validated against this project's known-working RT raster:

        RT = clip(DN - 1000, 0, 10000)

    The output stays UInt16 on a 0-10000 scale. The inference runner
    subsequently divides by 10000.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input raster does not exist: {input_path}"
        )

    print(f"Converting: {input_path.name}")

    with rasterio.open(input_path) as src:
        data = src.read(1)
        profile = src.profile.copy()

        print(
            "Input range:",
            int(data.min()),
            "to",
            int(data.max()),
        )

        converted = (
            data.astype(np.int32)
            - L2A_DN_OFFSET
        )

        converted = np.clip(
            converted,
            RT_MIN,
            RT_MAX,
        ).astype(np.uint16)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile.update(
        driver="GTiff",
        dtype="uint16",
        count=1,
        nodata=RT_NODATA,
        compress="lzw",
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:
        dst.write(converted, 1)
        dst.scales = [RT_SCALE]
        dst.offsets = [0.0]

    print(
        "RT output range:",
        int(converted.min()),
        "to",
        int(converted.max()),
    )
    print(f"Saved RT-style raster: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage:\n"
            "python sentinel_preprocess.py <INPUT_L2A_RASTER> <OUTPUT_RT_TIF>"
        )

    convert_l2a_to_rt(
        sys.argv[1],
        sys.argv[2],
    )
