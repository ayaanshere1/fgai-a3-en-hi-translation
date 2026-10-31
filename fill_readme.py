"""Write the auto-generated result tables (results/readme_sections/*.md) into README.md.

Usage (from the repository root, after copying the notebook outputs in):
    python fill_readme.py
Each section replaces the text between <!-- AUTO:name --> and <!-- /AUTO:name --> markers.
"""
import re
from pathlib import Path


def fill_readme(readme_path="README.md", sections_dir="results/readme_sections"):
    readme_path, sections_dir = Path(readme_path), Path(sections_dir)
    if not sections_dir.exists():
        raise SystemExit(f"{sections_dir} not found - run the notebook and copy results/ into the repo first.")
    text = readme_path.read_text(encoding="utf-8")
    filled = []
    for sec in sorted(sections_dir.glob("*.md")):
        body = sec.read_text(encoding="utf-8").strip()
        pat = re.compile(rf"(<!-- AUTO:{sec.stem} -->)(.*?)(<!-- /AUTO:{sec.stem} -->)", re.S)
        text, n = pat.subn(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)
        if n:
            filled.append(sec.stem)
    readme_path.write_text(text, encoding="utf-8")
    missing = sorted(set(re.findall(r"<!-- AUTO:(\w+) -->", text)) - set(filled))
    print("Filled:", ", ".join(filled) or "nothing")
    if missing:
        print("Still empty:", ", ".join(missing))


if __name__ == "__main__":
    fill_readme()
