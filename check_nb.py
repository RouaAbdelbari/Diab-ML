import nbformat
import sys

nb_path = r"c:\Users\TENPRO\Desktop\Diabetes\ml_optimization.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Print cell indices and some content to identify blocks
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        source = cell.source
        print(f"Cell {i} starts with: {source[:60]}")
