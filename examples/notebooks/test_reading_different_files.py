import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    from pivpy import io, pivpy, graphics

    return (io,)


@app.cell
def _():
    import xarray as xr
    from typing import List
    import numpy as np
    import pandas as pd
    from importlib.resources import files
    import matplotlib.pyplot as plt

    return files, np, pd, plt, xr, List


@app.cell
def _(files):
    import pathlib

    path = pathlib.Path(files("pivpy").joinpath("data"))
    subdirs = [x for x in sorted(path.glob('**/*')) if x.is_dir()]
    subdirs = [s for s in subdirs if s.stem != '.ipynb_checkpoints']

    test_files = []
    for d in subdirs:
        matches = [x for x in sorted(d.glob('[!.]*')) if not x.is_dir() ]
        test_files.append(matches[0])

    print(test_files)
    return (test_files,)


@app.cell
def _(io, plt, test_files):
    # Each subdirectory's first file isn't guaranteed to be PIV data (some
    # contain readmes/images alongside the actual datasets) -- skip whatever
    # doesn't parse as a supported format instead of aborting the whole scan.
    for file in test_files:
        try:
            variables, units, rows, cols, dt, frame, method = io.parse_header(file)
            print(file.stem, method)
            ds = io.read_piv(file)
        except (ValueError, OSError, ImportError) as e:
            print(f"{file.stem}: skipping ({e})")
            continue
        plt.figure()
        ds.isel(t=0).piv.quiver(arrScale=5)
        plt.title(file.stem)
    return


if __name__ == "__main__":
    app.run()
