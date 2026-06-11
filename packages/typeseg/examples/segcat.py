#!/usr/bin/env python3
"""segcat - colourised content-type segmentation in the terminal.

Thin wrapper kept for `python examples/segcat.py`. The implementation now
lives in the package (`typeseg._cli`) and ships as the `typeseg` / `segcat`
console command, so after `pip install typeseg` you can just run::

    typeseg file.html
    cat foo | typeseg --model fast
    typeseg --demo
"""
from typeseg._cli import main

if __name__ == "__main__":
    main()
