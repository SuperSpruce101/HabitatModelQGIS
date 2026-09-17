from pathlib import Path

import processing

from qgis.core import QgsApplication
from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon

from .habitat_algorithm import HabitatProcessingProvider
from .settings_dialog import HabitatSettingsDialog


class HabitatClassifier:

    def __init__(self, iface):
        self.iface = iface
        self.provider = None

        self.settings_action = None
        self.predict_action = None


    def initGui(self):

        # ----------------------------------------------------
        # REGISTER PROCESSING PROVIDER
        # ----------------------------------------------------

        self.provider = HabitatProcessingProvider()

        QgsApplication.processingRegistry().addProvider(
            self.provider
        )


        # ----------------------------------------------------
        # ICON
        # ----------------------------------------------------

        icon_path = (
            Path(__file__).resolve().parent
            / "icon.png"
        )

        icon = QIcon(str(icon_path))


        # ----------------------------------------------------
        # PREDICT HABITATS TOOLBAR BUTTON
        # ----------------------------------------------------

        self.predict_action = QAction(
            icon,
            "Predict Habitats",
            self.iface.mainWindow(),
        )

        self.predict_action.setToolTip(
            "Run Habitat Classifier"
        )

        self.predict_action.triggered.connect(
            self.open_prediction_tool
        )

        # Add to top QGIS toolbar
        self.iface.addToolBarIcon(
            self.predict_action
        )

        # Also add to plugin menu
        self.iface.addPluginToMenu(
            "&Habitat Classifier",
            self.predict_action,
        )


        # ----------------------------------------------------
        # SETTINGS ACTION
        # ----------------------------------------------------

        self.settings_action = QAction(
            "Habitat Classifier Settings",
            self.iface.mainWindow(),
        )

        self.settings_action.triggered.connect(
            self.open_settings
        )

        self.iface.addPluginToMenu(
            "&Habitat Classifier",
            self.settings_action,
        )


    def open_prediction_tool(self):

        processing.execAlgorithmDialog(
            "habitat_classifier:predict_habitat"
        )

    def open_settings(self):

        dialog = HabitatSettingsDialog(
            self.iface.mainWindow()
        )

        dialog.exec()


    def unload(self):

        # Remove Processing provider
        if self.provider is not None:

            QgsApplication.processingRegistry().removeProvider(
                self.provider
            )

        # Remove toolbar button
        if self.predict_action is not None:

            self.iface.removeToolBarIcon(
                self.predict_action
            )

            self.iface.removePluginMenu(
                "&Habitat Classifier",
                self.predict_action,
            )

        # Remove settings menu
        if self.settings_action is not None:

            self.iface.removePluginMenu(
                "&Habitat Classifier",
                self.settings_action,
            )
    