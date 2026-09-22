import subprocess
from pathlib import Path
from .external_environment import clean_external_environment

import processing

from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtGui import QColor, QIcon

from qgis.core import (
    QgsCoordinateTransform,
    QgsPalettedRasterRenderer,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterExtent,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingProvider,
    QgsProcessingUtils,
    QgsProject,
    QgsRasterLayer,
)

# ------------------------------------------------------------
# OUTPUT STYLING
# ------------------------------------------------------------

class HabitatStylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    def postProcessLayer(self, layer, context, feedback):
        classes = [
            QgsPalettedRasterRenderer.Class(1, QColor("#6A3D9A"), "Forest"),
            QgsPalettedRasterRenderer.Class(2, QColor("#2166AC"), "Shrubland"),
            QgsPalettedRasterRenderer.Class(3, QColor("#92C5DE"), "Grassland / Meadow"),
            QgsPalettedRasterRenderer.Class(4, QColor("#A8DDA8"), "Cryosphere"),
            QgsPalettedRasterRenderer.Class(5, QColor("#FFD92F"), "Bare Terrain"),
        ]

        renderer = QgsPalettedRasterRenderer(
            layer.dataProvider(),
            1,
            classes,
        )

        layer.setRenderer(renderer)
        layer.setName("Habitat Prediction")
        layer.triggerRepaint()


class ForestStylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    def postProcessLayer(self, layer, context, feedback):
        classes = [
            QgsPalettedRasterRenderer.Class(
                1,
                QColor("#A1D99B"),
                "Gentle / Plateau Forest",
            ),
            QgsPalettedRasterRenderer.Class(
                2,
                QColor("#238B45"),
                "Steep-Slope Forest",
            ),
            QgsPalettedRasterRenderer.Class(
                3,
                QColor("#3182BD"),
                "Valley Forest",
            ),
        ]

        renderer = QgsPalettedRasterRenderer(
            layer.dataProvider(),
            1,
            classes,
        )

        layer.setRenderer(renderer)
        layer.setName("Forest Terrain Subclasses")
        layer.triggerRepaint()


class DrainageStylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    def postProcessLayer(self, layer, context, feedback):
        classes = [
            QgsPalettedRasterRenderer.Class(
                1,
                QColor("#FF0000"),
                "Likely River / Drainage Network",
            ),
        ]

        renderer = QgsPalettedRasterRenderer(
            layer.dataProvider(),
            1,
            classes,
        )

        layer.setRenderer(renderer)
        layer.setName("Likely River / Drainage Network")
        layer.triggerRepaint()


# ------------------------------------------------------------
# HABITAT PREDICTION ALGORITHM
# ------------------------------------------------------------

class HabitatPredictionAlgorithm(QgsProcessingAlgorithm):
    B02 = "B02"
    B03 = "B03"
    B04 = "B04"
    DEM = "DEM"
    EXTENT = "EXTENT"

    # Geomorphological controls
    ENABLE_FLOW_REFINEMENT = "ENABLE_FLOW_REFINEMENT"
    FLOW_THRESHOLD = "FLOW_THRESHOLD"
    BARE_MIN_PROBABILITY = "BARE_MIN_PROBABILITY"
    FLOW_BOOST = "FLOW_BOOST"

    CREATE_DRAINAGE = "CREATE_DRAINAGE"
    DRAINAGE_THRESHOLD = "DRAINAGE_THRESHOLD"

    ENABLE_FOREST_SUBCLASSES = "ENABLE_FOREST_SUBCLASSES"
    SLOPE_THRESHOLD = "SLOPE_THRESHOLD"
    TPI_THRESHOLD = "TPI_THRESHOLD"
    TPI_WINDOW = "TPI_WINDOW"

    # Outputs
    OUTPUT_B02 = "OUTPUT_B02"
    OUTPUT_B03 = "OUTPUT_B03"
    OUTPUT_B04 = "OUTPUT_B04"
    OUTPUT_DEM = "OUTPUT_DEM"
    OUTPUT_DEM_ALIGNED = "OUTPUT_DEM_ALIGNED"
    OUTPUT_PREDICTION = "OUTPUT_PREDICTION"
    OUTPUT_FOREST_SUBCLASSES = "OUTPUT_FOREST_SUBCLASSES"
    OUTPUT_DRAINAGE = "OUTPUT_DRAINAGE"

    def name(self):
        return "predict_habitat"

    def displayName(self):
        return "Predict Habitats"

    def group(self):
        return "Habitat Mapping"

    def groupId(self):
        return "habitat_mapping"

    def shortHelpString(self):
        return (
            "Select Sentinel-2 L2A B02, B03 and B04 bands and a DEM, "
            "then draw a rectangular study area. Raw Sentinel-2 L2A "
            "JP2 bands are automatically converted to the RT-style "
            "0-10000 reflectance scale expected by the trained model. "
            "Already-prepared TIFF inputs are used as supplied. The "
            "selected rasters are clipped to the study area, the DEM is "
            "aligned to the Sentinel grid, and the habitat model is run. "
            "Optional drainage-network and forest-subclassification "
            "outputs can also be generated."
        )

    def createInstance(self):
        return HabitatPredictionAlgorithm()

    def initAlgorithm(self, config=None):
        # --------------------------------------------------------
        # INPUTS
        # --------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.B02,
                "Sentinel-2 B02 (Blue)",
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.B03,
                "Sentinel-2 B03 (Green)",
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.B04,
                "Sentinel-2 B04 (Red)",
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.DEM,
                "Digital Elevation Model",
            )
        )

        self.addParameter(
            QgsProcessingParameterExtent(
                self.EXTENT,
                "Study area",
            )
        )

        # --------------------------------------------------------
        # GEOMORPHOLOGICAL CONTROLS
        # --------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_FLOW_REFINEMENT,
                "Refine Bare Terrain using flow accumulation",
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.FLOW_THRESHOLD,
                "Normalised flow threshold (pilot default: 0.35)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.35,
                minValue=0.0,
                maxValue=1.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.BARE_MIN_PROBABILITY,
                "Minimum Bare Terrain probability (pilot default: 0.01)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.01,
                minValue=0.0,
                maxValue=1.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.FLOW_BOOST,
                "Flow influence / boost (pilot default: 2.0)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=2.0,
                minValue=0.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CREATE_DRAINAGE,
                "Create likely river / drainage network",
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.DRAINAGE_THRESHOLD,
                "River flow threshold (0–1)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.70,
                minValue=0.0,
                maxValue=1.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_FOREST_SUBCLASSES,
                "Create terrain-based Forest subclasses",
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.SLOPE_THRESHOLD,
                "Steep-Slope Forest threshold ° (pilot default: 45)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=45.0,
                minValue=0.0,
                maxValue=90.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.TPI_THRESHOLD,
                "Valley Forest TPI threshold (pilot default: -4.4)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=-4.4,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.TPI_WINDOW,
                "TPI neighbourhood size in pixels (pilot default: 9)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=9,
                minValue=3,
            )
        )

        # --------------------------------------------------------
        # OUTPUTS
        # --------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_PREDICTION,
                "Habitat Prediction",
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_DEM_ALIGNED,
                "Aligned DEM",
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_FOREST_SUBCLASSES,
                "Forest Terrain Subclasses",
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_DRAINAGE,
                "Likely River / Drainage Network",
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_B02,
                "Clipped B02",
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_B03,
                "Clipped B03",
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_B04,
                "Clipped B04",
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_DEM,
                "Clipped DEM",
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Keep post-processors alive until QGIS has loaded the results.
        self._post_processors = []

        # ========================================================
        # SHARED RUNTIME SETUP
        # ========================================================
        plugin_dir = Path(__file__).resolve().parent

        python_exe = QSettings().value(
            "HabitatModelQGIS/python_executable",
            "",
        )

        if not python_exe:
            raise QgsProcessingException(
                "No external Python environment has been configured. "
                "Open Habitat Classifier Settings first."
            )

        if not Path(python_exe).exists():
            raise QgsProcessingException(
                f"Configured Python executable does not exist:\n{python_exe}"
            )

        # Build the same isolated external-Python environment used by
        # the automatic setup dialog. This prevents QGIS / OSGeo4W DLLs
        # from interfering with PyTorch and the external geospatial stack.
        clean_env = clean_external_environment(
            python_exe
        )

        inference_script = plugin_dir / "inference_runner.py"
        sentinel_preprocess_script = plugin_dir / "sentinel_preprocess.py"
        forest_script = plugin_dir / "forest_subclasses.py"
        drainage_script = plugin_dir / "drainage_network.py"
        model_path = (
            plugin_dir
            / "models"
            / "baseline_RGB_only_64x64.pth"
        )

        if not inference_script.exists():
            raise QgsProcessingException(
                f"Inference script not found:\n{inference_script}"
            )

        if not model_path.exists():
            raise QgsProcessingException(
                f"Model checkpoint not found:\n{model_path}"
            )

        # ========================================================
        # GET INPUTS / SETTINGS
        # ========================================================
        b02 = self.parameterAsRasterLayer(parameters, self.B02, context)
        b03 = self.parameterAsRasterLayer(parameters, self.B03, context)
        b04 = self.parameterAsRasterLayer(parameters, self.B04, context)
        dem = self.parameterAsRasterLayer(parameters, self.DEM, context)

        enable_forest_subclasses = self.parameterAsBool(
            parameters,
            self.ENABLE_FOREST_SUBCLASSES,
            context,
        )
        slope_threshold = self.parameterAsDouble(
            parameters,
            self.SLOPE_THRESHOLD,
            context,
        )
        tpi_threshold = self.parameterAsDouble(
            parameters,
            self.TPI_THRESHOLD,
            context,
        )
        tpi_window = self.parameterAsInt(
            parameters,
            self.TPI_WINDOW,
            context,
        )

        create_drainage = self.parameterAsBool(
            parameters,
            self.CREATE_DRAINAGE,
            context,
        )
        drainage_threshold = self.parameterAsDouble(
            parameters,
            self.DRAINAGE_THRESHOLD,
            context,
        )

        enable_flow_refinement = self.parameterAsBool(
            parameters,
            self.ENABLE_FLOW_REFINEMENT,
            context,
        )
        flow_threshold = self.parameterAsDouble(
            parameters,
            self.FLOW_THRESHOLD,
            context,
        )
        bare_min_probability = self.parameterAsDouble(
            parameters,
            self.BARE_MIN_PROBABILITY,
            context,
        )
        flow_boost = self.parameterAsDouble(
            parameters,
            self.FLOW_BOOST,
            context,
        )

        if enable_flow_refinement:
            feedback.pushInfo(
                f"Bare Terrain flow refinement enabled: "
                f"threshold={flow_threshold}, "
                f"min bare probability={bare_min_probability}, "
                f"boost={flow_boost}"
            )

        # ========================================================
        # OUTPUT PATHS
        # ========================================================
        output_b02 = self.parameterAsOutputLayer(
            parameters,
            self.OUTPUT_B02,
            context,
        )
        output_b03 = self.parameterAsOutputLayer(
            parameters,
            self.OUTPUT_B03,
            context,
        )
        output_b04 = self.parameterAsOutputLayer(
            parameters,
            self.OUTPUT_B04,
            context,
        )
        output_dem = self.parameterAsOutputLayer(
            parameters,
            self.OUTPUT_DEM,
            context,
        )
        output_dem_aligned = self.parameterAsOutputLayer(
            parameters,
            self.OUTPUT_DEM_ALIGNED,
            context,
        )
        output_prediction = self.parameterAsOutputLayer(
            parameters,
            self.OUTPUT_PREDICTION,
            context,
        )
        output_forest_subclasses = self.parameterAsOutputLayer(
            parameters,
            self.OUTPUT_FOREST_SUBCLASSES,
            context,
        )
        output_drainage = self.parameterAsOutputLayer(
            parameters,
            self.OUTPUT_DRAINAGE,
            context,
        )

        # ========================================================
        # STUDY AREA
        # ========================================================
        extent = self.parameterAsExtent(
            parameters,
            self.EXTENT,
            context,
        )

        selected_extent_string = (
            f"{extent.xMinimum()},"
            f"{extent.xMaximum()},"
            f"{extent.yMinimum()},"
            f"{extent.yMaximum()}"
        )
        feedback.pushInfo(f"Selected study area: {selected_extent_string}")

        # ========================================================
        # CLIPPING HELPER
        # ========================================================
        def clip_raster(input_layer, output_path, name):
            feedback.pushInfo(f"Clipping {name}...")

            project_crs = QgsProject.instance().crs()
            raster_crs = input_layer.crs()

            feedback.pushInfo(f"{name} CRS: {raster_crs.authid()}")

            transform = QgsCoordinateTransform(
                project_crs,
                raster_crs,
                QgsProject.instance(),
            )
            raster_extent = transform.transformBoundingBox(extent)

            raster_extent_string = (
                f"{raster_extent.xMinimum()},"
                f"{raster_extent.xMaximum()},"
                f"{raster_extent.yMinimum()},"
                f"{raster_extent.yMaximum()}"
            )

            feedback.pushInfo(
                f"{name} clipping extent: {raster_extent_string}"
            )

            processing.run(
                "gdal:cliprasterbyextent",
                {
                    "INPUT": input_layer,
                    "PROJWIN": raster_extent_string,
                    "NODATA": None,
                    "OPTIONS": "",
                    "DATA_TYPE": 0,
                    "EXTRA": "",
                    "OUTPUT": output_path,
                },
                context=context,
                feedback=feedback,
            )

        # ========================================================
        # SENTINEL-2 L2A PREPROCESSING HELPERS
        # ========================================================

        def is_raw_l2a_jp2(input_layer):
            source = input_layer.source().split("|")[0]
            return Path(source).suffix.lower() == ".jp2"

        def convert_clipped_l2a_to_rt(input_path, output_path, name):
            if not sentinel_preprocess_script.exists():
                raise QgsProcessingException(
                    "Raw Sentinel-2 JP2 input was selected, but "
                    "sentinel_preprocess.py is missing from the plugin:\n"
                    f"{sentinel_preprocess_script}"
                )

            feedback.pushInfo(
                f"Converting clipped {name} from Sentinel-2 L2A DN "
                "to RT-style reflectance values..."
            )

            conversion_command = [
                python_exe,
                str(sentinel_preprocess_script),
                str(input_path),
                str(output_path),
            ]

            conversion_result = subprocess.run(
                conversion_command,
                capture_output=True,
                text=True,
                env=clean_env,
            )

            if conversion_result.stdout:
                for line in conversion_result.stdout.splitlines():
                    feedback.pushInfo(line)

            if conversion_result.returncode != 0:
                error_message = (
                    conversion_result.stderr
                    if conversion_result.stderr
                    else "Unknown Sentinel-2 preprocessing error."
                )
                raise QgsProcessingException(
                    f"{name} Sentinel-2 preprocessing failed:\n\n"
                    + error_message
                )

            if not Path(output_path).exists():
                raise QgsProcessingException(
                    f"{name} preprocessing finished, but no RT-style "
                    "GeoTIFF was created."
                )

        def prepare_sentinel_band(input_layer, output_path, name):
            """
            Raw JP2: clip first, then convert only the selected study area.
            Prepared TIFF: keep the existing workflow and clip directly.
            """
            if is_raw_l2a_jp2(input_layer):
                feedback.pushInfo(
                    f"{name}: raw Sentinel-2 L2A JP2 detected."
                )

                temporary_clip = QgsProcessingUtils.generateTempFilename(
                    f"{name}_raw_l2a_clip.tif"
                )

                clip_raster(
                    input_layer,
                    temporary_clip,
                    name,
                )

                convert_clipped_l2a_to_rt(
                    temporary_clip,
                    output_path,
                    name,
                )
            else:
                feedback.pushInfo(
                    f"{name}: prepared raster detected; "
                    "no L2A RT conversion applied."
                )

                clip_raster(
                    input_layer,
                    output_path,
                    name,
                )

        # ========================================================
        # CLIP + PREPARE INPUT RASTERS
        # ========================================================
        prepare_sentinel_band(b02, output_b02, "B02")
        prepare_sentinel_band(b03, output_b03, "B03")
        prepare_sentinel_band(b04, output_b04, "B04")

        clip_raster(dem, output_dem, "DEM")

        feedback.pushInfo(
            "Finished Sentinel preprocessing and study-area clipping."
        )

        # ========================================================
        # ALIGN DEM TO B04 GRID
        # ========================================================
        feedback.pushInfo("Aligning DEM to Sentinel B04 grid...")

        clipped_b04_layer = QgsRasterLayer(
            output_b04,
            "Clipped B04",
        )

        if not clipped_b04_layer.isValid():
            raise QgsProcessingException(
                "Could not load clipped B04 raster."
            )

        reference_extent = clipped_b04_layer.extent()
        reference_crs = clipped_b04_layer.crs()
        pixel_size_x = clipped_b04_layer.rasterUnitsPerPixelX()
        pixel_size_y = clipped_b04_layer.rasterUnitsPerPixelY()

        feedback.pushInfo(f"Reference CRS: {reference_crs.authid()}")
        feedback.pushInfo(
            f"Reference pixel size: {pixel_size_x} x {pixel_size_y}"
        )
        feedback.pushInfo(
            "Reference dimensions: "
            f"{clipped_b04_layer.width()} x {clipped_b04_layer.height()}"
        )

        reference_extent_string = (
            f"{reference_extent.xMinimum()},"
            f"{reference_extent.xMaximum()},"
            f"{reference_extent.yMinimum()},"
            f"{reference_extent.yMaximum()}"
        )

        processing.run(
            "gdal:warpreproject",
            {
                "INPUT": output_dem,
                "SOURCE_CRS": None,
                "TARGET_CRS": reference_crs,
                "RESAMPLING": 1,  # Bilinear for continuous DEM
                "NODATA": None,
                "TARGET_RESOLUTION": pixel_size_x,
                "OPTIONS": "",
                "DATA_TYPE": 6,  # Float32
                "TARGET_EXTENT": reference_extent_string,
                "TARGET_EXTENT_CRS": reference_crs,
                "MULTITHREADING": True,
                "EXTRA": "",
                "OUTPUT": output_dem_aligned,
            },
            context=context,
            feedback=feedback,
        )

        feedback.pushInfo("DEM alignment complete.")

        # ========================================================
        # HABITAT PREDICTION
        # ========================================================
        feedback.pushInfo("Starting habitat prediction...")

        command = [
            python_exe,
            str(inference_script),

            # Sentinel-2 inputs
            output_b02,
            output_b03,
            output_b04,

            # DEM already aligned to the Sentinel grid
            output_dem_aligned,

            # Model + output
            str(model_path),
            output_prediction,

            # Bare Terrain refinement controls
            "1" if enable_flow_refinement else "0",
            str(flow_threshold),
            str(bare_min_probability),
            str(flow_boost),
        ]

        feedback.pushInfo("Launching external ML environment...")
        feedback.pushInfo(f"Python: {python_exe}")

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=clean_env,
        )

        if result.stdout:
            for line in result.stdout.splitlines():
                feedback.pushInfo(line)

        if result.returncode != 0:
            error_message = (
                result.stderr
                if result.stderr
                else "Unknown inference error."
            )
            raise QgsProcessingException(
                "Habitat prediction failed:\n\n" + error_message
            )

        if not Path(output_prediction).exists():
            raise QgsProcessingException(
                "Habitat inference finished, but the prediction raster "
                "was not created."
            )

        feedback.pushInfo("Habitat prediction completed successfully.")

        # ========================================================
        # DRAINAGE NETWORK
        # ========================================================
        if create_drainage:
            if not drainage_script.exists():
                raise QgsProcessingException(
                    f"Drainage script not found:\n{drainage_script}"
                )

            if not output_drainage:
                raise QgsProcessingException(
                    "Drainage network is enabled, but no drainage output "
                    "destination was supplied."
                )

            feedback.pushInfo(
                "Creating likely river / drainage network..."
            )

            drainage_command = [
                python_exe,
                str(drainage_script),
                output_dem_aligned,
                output_drainage,
                str(drainage_threshold),
            ]

            drainage_result = subprocess.run(
                drainage_command,
                capture_output=True,
                text=True,
                env=clean_env,
            )

            if drainage_result.stdout:
                for line in drainage_result.stdout.splitlines():
                    feedback.pushInfo(line)

            if drainage_result.returncode != 0:
                error_message = (
                    drainage_result.stderr
                    if drainage_result.stderr
                    else "Unknown drainage-network error."
                )
                raise QgsProcessingException(
                    "Drainage network creation failed:\n\n"
                    + error_message
                )

            if not Path(output_drainage).exists():
                raise QgsProcessingException(
                    "Drainage processing finished but no output raster "
                    "was created."
                )

            feedback.pushInfo(
                "Drainage network created successfully."
            )

        # ========================================================
        # FOREST TERRAIN SUBCLASSIFICATION
        # ========================================================
        if enable_forest_subclasses:
            if not forest_script.exists():
                raise QgsProcessingException(
                    f"Forest subclassification script not found:\n"
                    f"{forest_script}"
                )

            if not output_forest_subclasses:
                raise QgsProcessingException(
                    "Forest subclassification is enabled, but no Forest "
                    "subclass output destination was supplied."
                )

            feedback.pushInfo(
                "Creating Forest terrain subclasses..."
            )

            forest_command = [
                python_exe,
                str(forest_script),
                output_prediction,
                output_dem_aligned,
                output_forest_subclasses,
                str(slope_threshold),
                str(tpi_threshold),
                str(tpi_window),
            ]

            forest_result = subprocess.run(
                forest_command,
                capture_output=True,
                text=True,
                env=clean_env,
            )

            if forest_result.stdout:
                for line in forest_result.stdout.splitlines():
                    feedback.pushInfo(line)

            if forest_result.returncode != 0:
                error_message = (
                    forest_result.stderr
                    if forest_result.stderr
                    else "Unknown Forest subclassification error."
                )
                raise QgsProcessingException(
                    "Forest subclassification failed:\n\n"
                    + error_message
                )

            if not Path(output_forest_subclasses).exists():
                raise QgsProcessingException(
                    "Forest subclassification finished but no output "
                    "raster was created."
                )

            feedback.pushInfo(
                "Forest terrain subclasses completed successfully."
            )


        # ========================================================
        # RESULTS
        # ========================================================
        results = {
            self.OUTPUT_B02: output_b02,
            self.OUTPUT_B03: output_b03,
            self.OUTPUT_B04: output_b04,
            self.OUTPUT_DEM: output_dem,
            self.OUTPUT_DEM_ALIGNED: output_dem_aligned,
            self.OUTPUT_PREDICTION: output_prediction,
        }

        if enable_forest_subclasses:
            results[self.OUTPUT_FOREST_SUBCLASSES] = (
                output_forest_subclasses
            )

        if create_drainage:
            results[self.OUTPUT_DRAINAGE] = output_drainage

        # ========================================================
        # APPLY OUTPUT STYLING
        # ========================================================
        habitat_details = context.layerToLoadOnCompletionDetails(
            output_prediction
        )
        habitat_post_processor = HabitatStylePostProcessor()
        habitat_details.setPostProcessor(habitat_post_processor)
        self._post_processors.append(habitat_post_processor)

        if enable_forest_subclasses:
            forest_details = context.layerToLoadOnCompletionDetails(
                output_forest_subclasses
            )
            forest_post_processor = ForestStylePostProcessor()
            forest_details.setPostProcessor(forest_post_processor)
            self._post_processors.append(forest_post_processor)

        if create_drainage:
            drainage_details = context.layerToLoadOnCompletionDetails(
                output_drainage
            )
            drainage_post_processor = DrainageStylePostProcessor()
            drainage_details.setPostProcessor(drainage_post_processor)
            self._post_processors.append(drainage_post_processor)

        return results


# ============================================================
# PROCESSING PROVIDER
# ============================================================

class HabitatProcessingProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(HabitatPredictionAlgorithm())

    def id(self):
        return "habitat_classifier"

    def name(self):
        return "Habitat Classifier"

    def longName(self):
        return self.name()

    def icon(self):
        icon_path = Path(__file__).resolve().parent / "icon.png"
        return QIcon(str(icon_path))