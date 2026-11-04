#!/usr/bin/env python3
"""Convert Jupyter notebooks in notebooks/ to blog posts in src/pages/posts/.

Usage:
    python scripts/notebook_to_post.py            # convert all notebooks
    python scripts/notebook_to_post.py my-nb.ipynb

Notebook front matter: put a raw cell at the very top of the notebook with
JSON metadata (title, description, date, tags). If absent, defaults are used.

The notebook is executed (so plots/outputs are fresh), then exported twice:
  * an HTML notebook version embedded as-is on a dedicated page, and
  * a Markdown post (classic reading version).

Each notebook produces up to two posts:
  * <stem>.md                -> /posts/<stem>            (classic markdown version)
  * <stem>-notebook.json     -> /notebooks/<stem>        (embedded nbconvert HTML)
The notebook version is published in two layouts ("full" and "centered")
under /notebooks/<stem>-full and /notebooks/<stem>-centered so both designs
can be compared side by side.
"""

import html
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
POSTS_DIR = ROOT / "src" / "pages" / "posts"
NB_HTML_DIR = ROOT / "src" / "notebook-html"

# Metadata in raw cell: title, description, date, tags.
DEFAULT_META = {
    "title": "Untitled Notebook",
    "description": "A notebook post.",
    "date": "2025-01-01",
    "tags": ["notebook"],
}

LAYOUTS = ["full", "centered"]


def extract_meta(nb):
    meta = dict(DEFAULT_META)
    cell = nb.get("cells", [])
    if cell and cell[0].get("cell_type") == "raw":
        try:
            source = cell[0].get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            parsed = json.loads(source)
            meta.update(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
    return meta


def run(cmd, cwd=ROOT):
    subprocess.run(cmd, check=True, cwd=cwd)


def export_html(nb_path, meta):
    """Execute the notebook and return the body HTML of the nbconvert HTML export."""
    run([
        "jupyter", "nbconvert",
        "--to", "html",
        "--execute",
        "--output", nb_path.stem + "-tmp",
        str(nb_path),
    ], cwd=NB_DIR)
    tmp_html = nb_path.with_name(nb_path.stem + "-tmp.html")
    if not tmp_html.exists():
        sys.exit(f"nbconvert did not produce {tmp_html}")
    text = tmp_html.read_text(encoding="utf-8")
    tmp_html.unlink()

    m = re.search(r'<main>(.*?)</main>', text, re.S)
    if not m:
        m = re.search(r'<div[^>]*id="notebook"[^>]*>(.*?)</body>', text, re.S)
    if not m:
        m = re.search(r'<body[^>]*>(.*?)</body>', text, re.S)
    if not m:
        sys.exit("Could not extract notebook body from nbconvert HTML output")
    body = m.group(1).strip()

    # Strip the metadata raw cell's JSON text from the HTML body. The raw
    # cell renders as leading text before the first jp-Cell div.
    body = re.sub(r'^.*?(?=<div class="jp-Cell)', '', body, count=1, flags=re.S)
    return body.strip()


def export_markdown(nb_path, meta):
    """Execute notebook in-place, export to Markdown, return the cleaned body."""
    run([
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        str(nb_path),
    ])
    run([
        "jupyter", "nbconvert",
        "--to", "markdown",
        str(nb_path),
    ], cwd=NB_DIR)
    converted = nb_path.with_suffix(".md")
    if not converted.exists():
        sys.exit(f"nbconvert did not produce {converted}")

    # Move generated image files into public/notebook-assets/<stem>/.
    assets_src = nb_path.with_name(nb_path.stem + "_files")
    assets_dir = ROOT / "public" / "notebook-assets" / nb_path.stem
    if assets_src.exists():
        assets_dir.mkdir(parents=True, exist_ok=True)
        for img in assets_src.iterdir():
            (assets_dir / img.name).write_bytes(img.read_bytes())
        body = converted.read_text(encoding="utf-8")
        body = body.replace(
            f"{nb_path.stem}_files/",
            f"/notebook-assets/{nb_path.stem}/",
        )
        converted.write_text(body, encoding="utf-8")
        for img in assets_src.iterdir():
            img.unlink()
        assets_src.rmdir()

    # Strip the metadata raw cell's JSON from the converted Markdown body.
    body = converted.read_text(encoding="utf-8")
    meta_json = json.dumps(meta, indent=2)
    for snippet in (meta_json, json.dumps(meta)):
        body = body.replace(snippet, "")
    # The page layout renders the title; omit the notebook's leading H1.
    return re.sub(r"\A\s*# [^\n]+\n+", "", body).lstrip("\n")


def front_matter(meta):
    tags = meta.get("tags", [])
    if isinstance(tags, list):
        tags = [str(t) for t in tags]
    else:
        tags = [str(tags)]
    return {
        "layout": "../../layouts/Post.astro",
        "title": str(meta["title"]),
        "description": str(meta["description"]),
        "date": str(meta["date"]),
        "tags": tags,
    }


def main():
    if not NB_DIR.exists():
        sys.exit(f"Notebook directory not found: {NB_DIR}")

    targets = [NB_DIR / name for name in sys.argv[1:]] or sorted(NB_DIR.glob("*.ipynb"))
    POSTS_DIR.mkdir(parents=True, exist_ok=True)
    NB_HTML_DIR.mkdir(parents=True, exist_ok=True)

    for nb_path in targets:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        meta = extract_meta(nb)
        fm = front_matter(meta)

        # Classic Markdown post.
        md_body = export_markdown(nb_path, meta)
        md_path = POSTS_DIR / (nb_path.stem + ".md")
        front = ["---"]
        front += [f"{k}: {json.dumps(v)}" for k, v in fm.items()]
        front += ["---", ""]
        md_path.write_text("\n".join(front) + md_body, encoding="utf-8")
        print(f"Converted {nb_path.name} -> {md_path}")

        # Embedded nbconvert HTML notebook, in both layouts.
        html_body = export_html(nb_path, meta)
        for layout in LAYOUTS:
            payload = dict(fm)
            payload["layout"] = layout
            payload["html"] = html_body
            out = NB_HTML_DIR / f"{nb_path.stem}-{layout}.json"
            out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            print(f"Converted {nb_path.name} -> {out} (layout: {layout})")


if __name__ == "__main__":
    main()
