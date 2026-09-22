import os
import shutil
import subprocess
from pathlib import Path

from .external_environment import (
    clean_external_environment,
    to_qprocess_environment,
)

from qgis.PyQt.QtCore import QProcess, QSettings
from qgis.PyQt.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
)


SETTINGS_KEY = "HabitatModelQGIS/python_executable"


class HabitatSettingsDialog(QDialog):
    """Configure or automatically create the external ML environment."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Habitat Classifier Settings")
        self.resize(720, 560)

        self.settings = QSettings()
        self.process = None
        self.setup_steps = []
        self.current_step = 0

        self.plugin_dir = Path(__file__).resolve().parent
        self.requirements_path = self.plugin_dir / "requirements.txt"

        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            self.environment_dir = Path(local_appdata) / "HabitatModelQGIS" / "ml-env"
        else:
            self.environment_dir = Path.home() / ".HabitatModelQGIS" / "ml-env"

        self.environment_python = self.environment_dir / "Scripts" / "python.exe"
        self.base_python = self._find_base_python()
        self.nvidia_detected = self._detect_nvidia_gpu()

        self._build_ui()
        self._refresh_status()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        intro = QLabel(
            "HabitatModelQGIS uses a separate Python environment for PyTorch "
            "and geospatial processing. You can create it automatically below, "
            "or choose an existing environment."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        current_group = QGroupBox("Current ML environment")
        current_layout = QVBoxLayout(current_group)

        path_row = QHBoxLayout()
        self.python_path_edit = QLineEdit()
        self.python_path_edit.setText(self.settings.value(SETTINGS_KEY, ""))
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_python)
        path_row.addWidget(self.python_path_edit, 1)
        path_row.addWidget(browse_button)
        current_layout.addLayout(path_row)

        action_row = QHBoxLayout()
        save_button = QPushButton("Save Selected Environment")
        save_button.clicked.connect(self._save_selected_environment)
        test_button = QPushButton("Test Environment")
        test_button.clicked.connect(self._test_selected_environment)
        action_row.addWidget(save_button)
        action_row.addWidget(test_button)
        action_row.addStretch()
        current_layout.addLayout(action_row)

        self.current_status_label = QLabel()
        self.current_status_label.setWordWrap(True)
        current_layout.addWidget(self.current_status_label)
        layout.addWidget(current_group)

        setup_group = QGroupBox("Automatic setup")
        setup_layout = QVBoxLayout(setup_group)

        base_text = str(self.base_python) if self.base_python else "No standalone Python installation detected"
        self.base_python_label = QLabel(f"Base Python: {base_text}")
        self.base_python_label.setWordWrap(True)
        setup_layout.addWidget(self.base_python_label)

        self.environment_label = QLabel(f"Environment location: {self.environment_dir}")
        self.environment_label.setWordWrap(True)
        setup_layout.addWidget(self.environment_label)

        gpu_text = "NVIDIA GPU detected" if self.nvidia_detected else "No NVIDIA GPU detected"
        self.gpu_detection_label = QLabel(f"Hardware detection: {gpu_text}")
        setup_layout.addWidget(self.gpu_detection_label)

        compute_row = QHBoxLayout()
        compute_row.addWidget(QLabel("PyTorch build:"))
        self.gpu_radio = QRadioButton("NVIDIA GPU (CUDA 12.8)")
        self.cpu_radio = QRadioButton("CPU")
        self.compute_group = QButtonGroup(self)
        self.compute_group.addButton(self.gpu_radio)
        self.compute_group.addButton(self.cpu_radio)
        if self.nvidia_detected:
            self.gpu_radio.setChecked(True)
        else:
            self.cpu_radio.setChecked(True)
        compute_row.addWidget(self.gpu_radio)
        compute_row.addWidget(self.cpu_radio)
        compute_row.addStretch()
        setup_layout.addLayout(compute_row)

        self.setup_button = QPushButton("Set Up Environment Automatically")
        self.setup_button.clicked.connect(self._start_automatic_setup)
        setup_layout.addWidget(self.setup_button)

        self.setup_status_label = QLabel("Ready.")
        self.setup_status_label.setWordWrap(True)
        setup_layout.addWidget(self.setup_status_label)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Installation progress will appear here.")
        setup_layout.addWidget(self.log_box, 1)
        layout.addWidget(setup_group, 1)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_row.addWidget(close_button)
        layout.addLayout(close_row)

    def _browse_python(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Python executable",
            "",
            "Python executable (python.exe);;All files (*)",
        )
        if path:
            self.python_path_edit.setText(path)

    def _save_selected_environment(self):
        python_exe = self.python_path_edit.text().strip()
        if not python_exe or not Path(python_exe).exists():
            QMessageBox.warning(self, "Habitat Classifier", "Choose a valid Python executable first.")
            return
        self.settings.setValue(SETTINGS_KEY, python_exe)
        self._refresh_status()
        QMessageBox.information(self, "Habitat Classifier", "External Python environment saved.")

    def _test_selected_environment(self):
        python_exe = self.python_path_edit.text().strip()
        if not python_exe or not Path(python_exe).exists():
            QMessageBox.warning(self, "Habitat Classifier", "Choose a valid Python executable first.")
            return

        clean_env = clean_external_environment(
            python_exe
        )

        code = (
            "import sys, numpy, scipy, rasterio, torch, torchvision, "
            "segmentation_models_pytorch, topotoolbox; "
            "print('Python:', sys.version.split()[0]); "
            "print('PyTorch:', torch.__version__); "
            "print('CUDA available:', torch.cuda.is_available()); "
            "print('Compute device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
        )

        try:
            result = subprocess.run(
                [python_exe, "-c", code],
                capture_output=True,
                text=True,
                timeout=60,
                env=clean_env,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Environment Test Failed", str(exc))
            return

        if result.returncode != 0:
            QMessageBox.critical(self, "Environment Test Failed", result.stderr or result.stdout)
            return

        QMessageBox.information(self, "Environment Test Successful", result.stdout.strip())

    def _refresh_status(self):
        if not hasattr(self, "current_status_label"):
            return
        saved = self.settings.value(SETTINGS_KEY, "")
        if saved and Path(saved).exists():
            self.current_status_label.setText(f"Status: configured\n{saved}")
        elif saved:
            self.current_status_label.setText("Status: configured path no longer exists.")
        else:
            self.current_status_label.setText("Status: no external ML environment configured.")

    def _start_automatic_setup(self):
        if self.process is not None:
            QMessageBox.information(self, "Habitat Classifier", "Environment setup is already running.")
            return

        if not self.base_python:
            QMessageBox.warning(
                self,
                "Python Not Found",
                "A compatible standalone Python installation could not be found.\n\n"
                "HabitatModelQGIS automatic setup requires a standalone "
                "Python 3.11 installation.\n\n"
                "Install Python 3.11 for Windows, reopen QGIS, and try again.\n\n"
                "The QGIS Python installation will not be modified.",
            )
            return

        if not self.requirements_path.exists():
            QMessageBox.critical(self, "Missing requirements.txt", f"Could not find:\n{self.requirements_path}")
            return

        self.environment_dir.parent.mkdir(parents=True, exist_ok=True)
        torch_index = (
            "https://download.pytorch.org/whl/cu128"
            if self.gpu_radio.isChecked()
            else "https://download.pytorch.org/whl/cpu"
        )

        self.setup_steps = [
            (
                "Creating Python virtual environment",
                [str(self.base_python), "-m", "venv", "--clear", str(self.environment_dir)],
            ),
            (
                "Upgrading pip",
                [str(self.environment_python), "-m", "pip", "install", "--upgrade", "pip"],
            ),
            (
                "Installing PyTorch",
                [
                    str(self.environment_python),
                    "-m",
                    "pip",
                    "install",
                    "torch==2.11.0",
                    "torchvision==0.26.0",
                    "--index-url",
                    torch_index,
                ],
            ),
            (
                "Installing HabitatModelQGIS dependencies",
                [
                    str(self.environment_python),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(self.requirements_path),
                ],
            ),
            (
                "Verifying installation",
                [
                    str(self.environment_python),
                    "-c",
                    (
                        "import numpy, scipy, rasterio, torch, torchvision, "
                        "segmentation_models_pytorch, topotoolbox; "
                        "print('Python environment OK'); "
                        "print('PyTorch:', torch.__version__); "
                        "print('CUDA available:', torch.cuda.is_available()); "
                        "print('Compute device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
                    ),
                ],
            ),
        ]

        self.current_step = 0
        self.log_box.clear()
        self._append_log("HabitatModelQGIS automatic environment setup")
        self._append_log(f"Base Python: {self.base_python}")
        self._append_log(f"Environment: {self.environment_dir}")
        self._append_log("PyTorch build: " + ("CUDA 12.8" if self.gpu_radio.isChecked() else "CPU"))
        self._append_log("")

        self.setup_button.setEnabled(False)
        self.gpu_radio.setEnabled(False)
        self.cpu_radio.setEnabled(False)
        self._run_next_step()

    def _run_next_step(self):
        if self.current_step >= len(self.setup_steps):
            self._setup_finished_successfully()
            return

        label, command = self.setup_steps[self.current_step]
        self.setup_status_label.setText(f"{label}...")
        self._append_log(f"\n[{self.current_step + 1}/{len(self.setup_steps)}] {label}")

        self.process = QProcess(self)

        # Run setup commands outside QGIS's Python/DLL environment.
        # Once the virtual environment exists, put its Python environment
        # first in PATH for all venv-based installation/verification steps.
        program_path = Path(command[0])
        is_venv_python = (
            program_path.name.lower() == "python.exe"
            and self.environment_dir in program_path.parents
        )

        if is_venv_python:
            clean_env = clean_external_environment(
                str(self.environment_python)
            )
        else:
            clean_env = clean_external_environment()

        self.process.setProcessEnvironment(
            to_qprocess_environment(clean_env)
        )
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)

        program = command[0]
        arguments = command[1:]
        self.process.start(program, arguments)

        if not self.process.waitForStarted(5000):
            self._setup_failed(f"Could not start:\n{program}")

    def _read_process_output(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self._append_log(data.rstrip())

    def _process_finished(self, exit_code, exit_status):
        if self.process is None:
            return

        self._read_process_output()
        process = self.process
        self.process = None
        process.deleteLater()

        if exit_status != QProcess.NormalExit or exit_code != 0:
            log_text = self.log_box.toPlainText()
            message = f"Setup step failed with exit code {exit_code}."
            if "WinError 1114" in log_text or "c10.dll" in log_text:
                message += (
                    "\n\nPyTorch could not initialise a Windows DLL. "
                    "The setup now isolates the ML environment from QGIS DLL paths. "
                    "If this still occurs after retrying, repair/install the "
                    "Microsoft Visual C++ 2015-2022 Redistributable (x64) and "
                    "update the NVIDIA driver, then run automatic setup again."
                )
            self._setup_failed(message)
            return

        self.current_step += 1
        self._run_next_step()

    def _setup_finished_successfully(self):
        python_exe = str(self.environment_python)
        self.settings.setValue(SETTINGS_KEY, python_exe)
        self.python_path_edit.setText(python_exe)
        self.setup_status_label.setText("Setup complete. The new environment is now selected.")
        self._append_log("\nSetup complete.")
        self._append_log(f"Saved Python executable: {python_exe}")
        self.setup_button.setEnabled(True)
        self.gpu_radio.setEnabled(True)
        self.cpu_radio.setEnabled(True)
        self._refresh_status()
        QMessageBox.information(
            self,
            "Habitat Classifier",
            "Machine-learning environment setup completed successfully.\n\n"
            "The plugin is now configured to use it.",
        )

    def _setup_failed(self, message):
        self.setup_status_label.setText("Setup failed. See the installation log below.")
        self._append_log(f"\nERROR: {message}")
        if self.process is not None:
            self.process.kill()
            self.process.deleteLater()
            self.process = None
        self.setup_button.setEnabled(True)
        self.gpu_radio.setEnabled(True)
        self.cpu_radio.setEnabled(True)
        QMessageBox.critical(
            self,
            "Automatic Setup Failed",
            message + "\n\nThe existing QGIS installation has not been modified.",
        )

    def _append_log(self, text):
        self.log_box.append(text)
        scrollbar = self.log_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _find_base_python(self):
        """
        Find a standalone Python version with binary-wheel support for the
        plugin dependencies.

        Use CPython 3.11 for the automatic ML environment.

        This is the version used by the project's known-working Windows
        environment with PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128,
        CUDA 12.8 and TopoToolbox 0.0.12. Other Python versions can still
        be selected manually, but automatic setup deliberately targets
        Python 3.11 for reproducibility.
        """
        clean_env = clean_external_environment()

        supported_versions = [(3, 11)]
        candidates = []

        # First ask the Windows Python launcher for the exact supported
        # versions. This avoids accidentally selecting Python 3.14 simply
        # because it is the newest/default interpreter.
        py_launcher = shutil.which("py")
        if py_launcher:
            for major, minor in supported_versions:
                try:
                    result = subprocess.run(
                        [
                            py_launcher,
                            f"-{major}.{minor}",
                            "-c",
                            "import sys; print(sys.executable)",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        env=clean_env,
                    )
                    if result.returncode == 0:
                        resolved = Path(result.stdout.strip())
                        if resolved.exists() and "qgis" not in str(resolved).lower():
                            candidates.append(resolved)
                except Exception:
                    pass

        # Then inspect interpreters that may be on PATH or in common
        # Windows installation locations.
        for command in ("python", "python3"):
            found = shutil.which(command)
            if found:
                candidates.append(Path(found))

        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            root = Path(local_appdata)
            candidates.extend((root / "Programs" / "Python").glob("Python*/python.exe"))
            candidates.extend((root / "Python").glob("pythoncore-*/python.exe"))

        for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
            program_files = os.environ.get(env_name)
            if program_files:
                candidates.extend(Path(program_files).glob("Python*/python.exe"))

        # Query each unique candidate and keep only versions for which the
        # plugin has a reliable binary-install path.
        compatible = []
        seen = set()

        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate

            key = str(resolved).lower()
            if key in seen or "qgis" in key:
                continue
            seen.add(key)

            if not resolved.exists():
                continue

            try:
                result = subprocess.run(
                    [
                        str(resolved),
                        "-c",
                        (
                            "import sys; "
                            "print(sys.executable); "
                            "print(sys.version_info.major); "
                            "print(sys.version_info.minor)"
                        ),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=clean_env,
                )
            except Exception:
                continue

            if result.returncode != 0:
                continue

            lines = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip()
            ]

            if len(lines) < 3:
                continue

            try:
                version = (int(lines[1]), int(lines[2]))
            except ValueError:
                continue

            if version not in supported_versions:
                continue

            actual_python = Path(lines[0])
            if actual_python.exists():
                compatible.append((version, actual_python))

        # Automatic setup deliberately targets Python 3.11.
        for preferred in supported_versions:
            for version, python_path in compatible:
                if version == preferred:
                    return python_path

        return None

    def _detect_nvidia_gpu(self):
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            default_path = Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")
            if default_path.exists():
                nvidia_smi = str(default_path)

        if not nvidia_smi:
            return False

        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                env=clean_external_environment(),
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False

    def closeEvent(self, event):
        if self.process is not None:
            answer = QMessageBox.question(
                self,
                "Setup In Progress",
                "Environment setup is still running. Stop it and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.process.kill()
            self.process = None
        event.accept()