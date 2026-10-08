---
title: Command Line
description: Inspect mixed text directly in your terminal with typeseg or segcat.
---
`pip install typeseg` installs the **`typeseg`** command and its alias **`segcat`**. See the content types inside a file directly in your terminal, with colored text, a legend, and a table of segment ranges and confidence scores.

## Inspect a file

Start with U-Net for fast local inspection:

```bash
typeseg --model fast file.txt
```

Use Mamba when full-file context matters more than throughput:

```bash
typeseg --model precise file.txt
```

`typeseg file.txt` also uses Mamba: **`precise` is the CLI default**. U-Net is selected explicitly with `--model fast`.

## Read from standard input

Omit the filename to read piped text:

```bash
cat file.txt | typeseg --model fast
```

In PowerShell:

```powershell
Get-Content -Raw file.txt | typeseg --model fast
```

The file argument is read as UTF-8, with replacement for invalid characters. For other encodings, decode the input first and retain the original byte mapping if you need byte-accurate annotations.

## Demo, alias, and module invocation

```bash
typeseg --model fast --demo       # built-in mixed-text example
segcat --model fast file.txt      # alias of typeseg
python -m typeseg --model fast file.txt
typeseg --help
```

## Understand the display

The output shows the active backend, input length, a content-type legend, highlighted source, and a table of **character ranges**, labels, and confidence. Range ends are excluded. Confidence describes content type, not maliciousness.

Text tinting is automatic in an interactive terminal. `NO_COLOR` disables automatic text tinting; `TYPESEG_COLOR=always` or `TYPESEG_COLOR=never` overrides it. The CLI also uses ANSI formatting for its legend and table.

For structured results and configurable post-processing, use the [Python API](../../getting-started/quick-start/). For the package reference, see [TypeSeg on PyPI](https://pypi.org/project/typeseg/).
