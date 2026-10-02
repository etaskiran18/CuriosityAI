# Philosophy corpus folder

This folder is for **real primary philosophical texts**, preferably public-domain `.md` or `.txt` files with metadata front matter.

In v3, the tiny generated seed notes were moved to `data/seed_examples/` so they do not pretend to be real philosophical readings.

## Recommended real-corpus workflow

Run:

```bash
python scripts/download_real_corpus.py --clear-existing
```

This downloads selected public-domain Project Gutenberg texts listed in:

```text
data/real_corpus_manifest.json
```

The downloader preserves source metadata and the Project Gutenberg text/license material in the downloaded file. Verify local copyright/public-domain status for your country before redistribution.

## Manual format

```md
---
title: The Republic
author: Plato
year: c. 375 BCE
tradition: ancient Greek philosophy
source_url: https://www.gutenberg.org/ebooks/1497
license: Project Gutenberg / public domain in many jurisdictions; verify local law
---

Text here...
```
