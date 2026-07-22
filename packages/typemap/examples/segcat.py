#!/usr/bin/env python3
"""segcat - colourised content-type segmentation in the terminal.

Thin wrapper kept for `python examples/segcat.py`. The implementation now
lives in the package (`typemap._cli`) and ships as the `typemap` / `segcat`
console command, so after `pip install typemap` you can just run::

    typemap file.html
    cat foo | typemap --model fast
    typemap --demo
"""
from typemap._cli import main

if __name__ == "__main__":
    main()
