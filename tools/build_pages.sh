#!/usr/bin/env bash
# Build the static, in-browser (Pyodide) version of the Panel app.
#
# Usage: tools/build_pages.sh [OUT_DIR]   (default: _site)
#
# Needs `build`, `panel` and nuosclab itself (`pip install ".[plot]"`; panel
# convert imports the app to check it) in the active environment. The pinned
# panel version decides which bokeh/panel wheels the page loads from the
# HoloViz CDN, so keep it in step with .github/workflows/pages.yml.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
out=$(mkdir -p "${1:-_site}" && cd "${1:-_site}" && pwd)

rm -rf "${out:?}"/*
python -m build --wheel --outdir "$out" "$repo"
wheel=$(cd "$out" && ls nuosclab-*.whl)

# panel convert resolves local wheels relative to the working directory and
# packs them into <name>.resources.zip next to the page.
cd "$out"
panel convert "$repo/tools/panel_app.py" \
    --to pyodide-worker \
    --out "$out" \
    --title nuosclab \
    --requirements numpy "./$wheel"

mv panel_app.html index.html
rm -f "$wheel" .gitignore
touch .nojekyll
echo "Built $out (serve with: python -m http.server -d $out)"
