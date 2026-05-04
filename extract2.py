import nbformat

nb_path = r"c:\Users\TENPRO\Desktop\Diabetes\ml_optimization.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

with open("cell16.py", "w", encoding="utf-8") as f:
    if len(nb.cells) > 16:
        f.write(nb.cells[16].source)
