import os
from pathlib import Path

package_name = "mongodb_connect2"

list_of_files = [
   ".github/workflows/ci.yaml",
   "src/__init__.py",
   "src/models/__init__.py", 
   "src/metrics/__init__.py",
   "src/utils/__init__.py",
   "src/Datasets/__init__.py",
   "src/Defence/__init__.py",
   "src/Defence/DifferentialPrivacy/__init__.py",
   "src/Defence/DifferentialPrivacy/model.py",
   "src/Defence/EdgePerturbation/__init__.py",
   "src/Defence/EdgePerturbation/model.py",
    "main.py",
    "src/attack/__init__.py",
    "src/attack/LOGRA/__init__.py",
    "src/attack/LOGRA/logra.py",
    "src/attack/CSGRA/__init__.py",
    "src/attack/CSGRA/csgra.py",
   "setup.py", 
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass # create an empty file