import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    """ tests pivpy.pivpy methods """
    import pathlib
    import numpy as np
    from importlib.resources import files
    from pivpy import io
    import pivpy.pivpy  # noqa: F401 -- registers the .piv accessor

    FILE1 = "Run000001.T000.D000.P000.H001.L.vec"
    FILE2 = "Run000002.T000.D000.P000.H001.L.vec"
    path = pathlib.Path(files("pivpy").joinpath("data"))
    path = path / "Insight"

    a = io.load_vec(path / FILE1)
    b = io.load_vec(path / FILE2)
    return io, np, a, b


@app.cell
def _(io):
    data = io.create_sample_field(rows=3, cols=3, noise_sigma=0.0)
    return (data,)


@app.cell
def _(data):
    data.piv.set_scale(1/16)
    return


@app.cell
def _(data):
    data
    return


@app.cell
def _(data):
    data.isel(t=0)["u"]
    return


@app.cell
def _(data):
    data["u"].differentiate("x")
    return


@app.cell
def _(data):
    data["v"].differentiate("y")
    return


@app.cell
def _(data):
    data.piv.strain()["w"]
    return


@app.cell
def _(io, np):
    def test_strain():
        """tests shear estimate"""
        data = io.create_sample_field(rows=2, cols=2, noise_sigma=0.0)
        data = data.piv.strain()
        print(data["u"])
        print(data["v"])
        print(data["w"])
        assert np.allclose(data["w"].values, 0.11328125, 1e-6)

    return (test_strain,)


@app.cell
def _(test_strain):
    test_strain()
    return


if __name__ == "__main__":
    app.run()
