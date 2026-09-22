import os
from pathlib import Path

from qgis.PyQt.QtCore import QProcessEnvironment


QGIS_ENV_VARS = (
    "PYTHONHOME",
    "PYTHONPATH",
    "QGIS_PREFIX_PATH",
    "QT_PLUGIN_PATH",
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    "GDAL_DATA",
    "GDAL_DRIVER_PATH",
    "PROJ_LIB",
    "PROJ_DATA",
    "OSGEO4W_ROOT",
)


def clean_external_environment(python_exe=None):
    """
    Build a clean environment for external standalone Python.

    Prevents QGIS / OSGeo4W DLLs from interfering with
    PyTorch, Rasterio and other packages in the plugin ML env.
    """

    env = os.environ.copy()

    # Remove variables injected by QGIS
    for name in QGIS_ENV_VARS:
        env.pop(name, None)

    # Remove QGIS / OSGeo4W directories from PATH
    filtered_path = []

    for entry in env.get("PATH", "").split(os.pathsep):

        entry = entry.strip().strip('"')

        if not entry:
            continue

        lower = entry.lower()

        if "qgis" in lower or "osgeo4w" in lower:
            continue

        filtered_path.append(entry)

    prepend = []

    if python_exe:
        python_exe = Path(python_exe)

        # .../ml-env/Scripts/python.exe
        scripts_dir = python_exe.parent
        venv_dir = scripts_dir.parent

        prepend.extend(
            [
                str(scripts_dir),
                str(venv_dir),
            ]
        )

        env["VIRTUAL_ENV"] = str(venv_dir)

    # Keep essential Windows DLL locations
    system_root = (
        env.get("SystemRoot")
        or env.get("WINDIR")
        or r"C:\Windows"
    )

    essentials = [
        str(Path(system_root) / "System32"),
        str(Path(system_root)),
    ]

    # De-duplicate PATH
    final_path = []
    seen = set()

    for entry in prepend + essentials + filtered_path:

        key = entry.lower()

        if key not in seen:
            seen.add(key)
            final_path.append(entry)

    env["PATH"] = os.pathsep.join(final_path)

    return env


def to_qprocess_environment(env):
    """
    Convert a normal environment dictionary to QProcessEnvironment.
    """

    qenv = QProcessEnvironment()

    for key, value in env.items():
        qenv.insert(
            str(key),
            str(value),
        )

    return qenv