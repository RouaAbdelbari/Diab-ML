import nbformat

nb_path = r"c:\Users\TENPRO\Desktop\Diabetes\ml_optimization.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

with open("cell2.py", "w", encoding="utf-8") as f:
    f.write(nb.cells[2].source)
with open("cell4.py", "w", encoding="utf-8") as f:
    f.write(nb.cells[4].source)
with open("cell6.py", "w", encoding="utf-8") as f:
    f.write(nb.cells[6].source)
with open("cell8.py", "w", encoding="utf-8") as f:
    f.write(nb.cells[8].source)
with open("cell14.py", "w", encoding="utf-8") as f:
    f.write(nb.cells[14].source)
