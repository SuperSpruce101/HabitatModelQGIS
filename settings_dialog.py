from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
)

import subprocess
import os


SETTINGS_KEY = "HabitatModelQGIS/python_executable"


class HabitatSettingsDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle(
            "Habitat Classifier Settings"
        )

        self.resize(
            600,
            140,
        )

        layout = QVBoxLayout()

        # ----------------------------------------------------
        # Python executable
        # ----------------------------------------------------

        layout.addWidget(
            QLabel(
                "Python executable containing PyTorch:"
            )
        )

        path_layout = QHBoxLayout()

        self.python_path = QLineEdit()

        browse_button = QPushButton(
            "Browse..."
        )

        browse_button.clicked.connect(
            self.browse_python
        )

        path_layout.addWidget(
            self.python_path
        )

        path_layout.addWidget(
            browse_button
        )

        layout.addLayout(
            path_layout
        )

        # ----------------------------------------------------
        # TEST BUTTON
        # ----------------------------------------------------

        test_button = QPushButton(
            "Test Environment"
        )

        test_button.clicked.connect(
            self.test_environment
        )

        layout.addWidget(
            test_button
        )

        # ----------------------------------------------------
        # SAVE BUTTON
        # ----------------------------------------------------

        save_button = QPushButton(
            "Save"
        )

        save_button.clicked.connect(
            self.save_settings
        )

        layout.addWidget(
            save_button
        )

        self.setLayout(
            layout
        )

        self.load_settings()


    # ========================================================
    # LOAD EXISTING SETTINGS
    # ========================================================

    def load_settings(self):

        settings = QSettings()

        saved_path = settings.value(
            SETTINGS_KEY,
            "",
        )

        self.python_path.setText(
            saved_path
        )


    # ========================================================
    # BROWSE
    # ========================================================

    def browse_python(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Python executable",
            "",
            "Python executable (python.exe)",
        )

        if filename:
            self.python_path.setText(
                filename
            )


    # ========================================================
    # TEST ENVIRONMENT
    # ========================================================

    def test_environment(self):

        python_exe = (
            self.python_path
            .text()
            .strip()
        )

        if not python_exe:
            QMessageBox.warning(
                self,
                "No Python selected",
                "Please select a Python executable.",
            )
            return

        command = [
            python_exe,
            "-c",
            (
                "import torch; "
                "import segmentation_models_pytorch as smp; "
                "import rasterio; "
                "import numpy; "
                "import scipy; "
                "print(torch.__version__)"
            ),
        ]

        try:

            clean_env = os.environ.copy()

            # Remove QGIS/Python environment variables that can interfere
            clean_env.pop("PYTHONHOME", None)
            clean_env.pop("PYTHONPATH", None)

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                env=clean_env,
            )

            if result.returncode == 0:

                QMessageBox.information(
                    self,
                    "Environment OK",
                    (
                        "Required packages found.\n\n"
                        "PyTorch version:\n"
                        f"{result.stdout.strip()}"
                    ),
                )

            else:

                QMessageBox.critical(
                    self,
                    "Environment Error",
                    result.stderr,
                )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Environment Error",
                str(error),
            )


    # ========================================================
    # SAVE
    # ========================================================

    def save_settings(self):

        python_exe = (
            self.python_path
            .text()
            .strip()
        )

        settings = QSettings()

        settings.setValue(
            SETTINGS_KEY,
            python_exe,
        )

        QMessageBox.information(
            self,
            "Settings Saved",
            "Python environment saved successfully.",
        )

        self.accept()