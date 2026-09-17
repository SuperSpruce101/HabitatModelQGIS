# HabitatModelQGIS

A QGIS plugin for machine-learning-assisted habitat classification using Sentinel-2 imagery and geomorphological information.

The plugin uses a U-Net segmentation model with a ResNet34 encoder to predict five broad habitat classes from Sentinel-2 RGB imagery. Optional terrain-based processing can also be used to refine Bare Terrain predictions, identify likely drainage networks, and subdivide Forest habitat using terrain characteristics.

## Habitat Classes

1. Forest
2. Shrubland
3. Grassland / Meadow
4. Cryosphere
5. Bare Terrain

## Features

- Sentinel-2 B02, B03 and B04 input
- Automatic preprocessing of raw Sentinel-2 L2A JP2 imagery
- U-Net habitat classification
- Automatic GPU acceleration when CUDA is available
- CPU fallback when CUDA is unavailable
- DEM alignment to the Sentinel-2 grid
- Flow-accumulation refinement of Bare Terrain
- Optional drainage-network generation
- Optional terrain-based Forest subclassification
- GeoTIFF outputs that can be loaded directly into QGIS

## Requirements

QGIS 3.x is required.

The machine-learning components run in a separate Python environment configured through the plugin.

Python dependencies are listed in:

```text
requirements.txt
