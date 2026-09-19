"""Genera los notebooks de 08_experimentation a partir de nb_src/*.py (formato percent:
`# %%` código, `# %% [markdown]` texto). Idempotente. Uso: uv run python 08_experimentation/build_nb.py
Para ejecutar: uv run jupyter nbconvert --to notebook --execute --inplace 08_experimentation/0X_*.ipynb
No sobreescribe un notebook que ya tiene outputs ejecutados salvo con `--force`.
"""
import re
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
for src in sorted((HERE / "nb_src").glob("*.py")):
    cells = []
    for block in re.split(r"^# %%", src.read_text(), flags=re.M)[1:]:
        header, _, body = block.partition("\n")
        body = body.strip("\n")
        if "[markdown]" in header:
            body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            cells.append(nbf.v4.new_markdown_cell(body))
        elif body:
            cells.append(nbf.v4.new_code_cell(body))
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    out = HERE / f"{src.stem}.ipynb"
    if out.exists() and "--force" not in sys.argv and any(c.get("outputs") for c in nbf.read(out, 4).cells):
        print("↷", out.name, "ya está ejecutado; usar --force para regenerarlo")
        continue
    nbf.write(nb, out)
    print("→", out.name, len(cells), "celdas")
