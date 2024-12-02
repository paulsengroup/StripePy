<!--
Copyright (C) 2024 Roberto Rossini <roberros@uio.no>

SPDX-License-Identifier: MIT
-->

# How to build StripePy's documentation

```bash
# Create a venv and install build requirements
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# Activate venv
. venv/bin/activate

# Clean old build files (optional)
make clean

make html
make latexpdf

make linkcheck
```

Open the HTML documentation:

```bash
# Linux
xdg-open _build/html/index.html

# macOS
open _build/html/index.html
```

Open the PDF documentation:

```bash
# Linux
xdg-open _build/latex/stripepy.pdf

# macOS
open _build/latex/stripepy.pdf
```
