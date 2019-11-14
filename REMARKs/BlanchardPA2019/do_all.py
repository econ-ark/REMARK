import os

os.system("jupytext --to py Code/Python/BlanchardPA2019.ipynb")

import importlib.util
spec = importlib.util.spec_from_file_location("Blanchard2019", "Code/Python/BlanchardPA2019.py")
Blanchard2019 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Blanchard2019)

