"""Structure checks for the lesson notebooks (they are not executed here: some train for minutes)."""
import ast
from pathlib import Path

import nbformat
import pytest

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS = sorted(NB_DIR.glob("[0-9][0-9]_*.ipynb"))


def test_all_eleven_lessons_present_and_numbered():
    assert [p.name[:2] for p in NOTEBOOKS] == [f"{i:02d}" for i in range(11)]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_is_valid_lesson(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    first = nb.cells[0]
    assert first.cell_type == "markdown" and first.source.startswith(f"# Lesson {int(path.name[:2])} ")
    assert "By the end of this lesson" in first.source
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    if path.name[:2] != "10":  # the wrap-up lesson has a quiz instead
        assert "Key takeaways" in text and "Exercises" in text and "Next" in text


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_code_cells_compile_and_have_no_saved_errors(path):
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = "\n".join(l for l in cell.source.splitlines() if not l.lstrip().startswith(("%", "!")))
        ast.parse(src)
        assert not [o for o in cell.outputs if o.output_type == "error"], f"saved error in {path.name}"


def test_lessons_do_not_use_removed_scripts():
    for path in NOTEBOOKS:
        raw = path.read_text(encoding="utf-8")
        for stale in ("simple_train", "preprocess.py", "simple_eval", "sequences.pkl"):
            assert stale not in raw, f"{path.name} mentions {stale}"
