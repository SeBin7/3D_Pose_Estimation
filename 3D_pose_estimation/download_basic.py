from pathlib import Path
from notebook_utils import collect_telemetry
import requests

if not Path("notebook_utils.py").exists():
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    with open("notebook_utils.py", "w") as f:
        f.write(r.text)

if not Path("engine3js.py").exists():
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/engine3js.py",
    )
    with open("engine3js.py", "w") as f:
        f.write(r.text)

collect_telemetry("3D-pose-estimation.ipynb")
