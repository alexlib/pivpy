import pathlib
import numpy as np
import pytest
import xarray as xr

from pivpy import io
from pivpy.compute_funcs import operf, spec2f, specf, ssf, tempspecf, vsf


def _pivpy_data_path() -> pathlib.Path:
    # Mirror the compatibility approach used in tests/test_io.py
    try:
        from importlib.resources import files

        return files("pivpy") / "data"
    except Exception:  # pragma: no cover
        from importlib.resources import path as resource_path

        with resource_path("pivpy", "data") as p:
            return pathlib.Path(p) / "data"


def test_loadpivtxt_parses_header_and_grid(tmp_path: pathlib.Path):
    p = tmp_path / "export.txt"
    p.write_text(
        "# some header\n"
        "Title: demo\n"
        "0 0 1 2\n"
        "1 0 3 4\n"
        "0 1 5 6\n"
        "1 1 7 8\n"
    )

    ds = io.loadpivtxt(p)
    assert isinstance(ds, xr.Dataset)
    assert set(["u", "v", "chc"]).issubset(ds.data_vars)
    assert ds["u"].shape == (2, 2, 1)
    assert "Attributes" in ds.attrs
    assert "some header" in ds.attrs["Attributes"]


def test_readsetfile_parses_values_and_lookup(tmp_path: pathlib.Path):
    p = tmp_path / "test.set"
    p.write_text(
        "delta_t = 0.01\n"
        "FrameDt: 2\n"
        "Name = \"Example\"\n"
    )

    attrs = io.readsetfile(p)
    assert attrs["delta_t"] == 0.01
    assert attrs["FrameDt"] == 2
    assert attrs["Name"] == "Example"

    assert io.readsetfile(p, "delta_t") == 0.01
    assert io.readsetfile(p, "frame_dt") == 2


def test_getattribute_reads_setfile(tmp_path: pathlib.Path):
    p = tmp_path / "test.exp"
    p.write_text("delta_t=0.02\n")

    all_attrs = io.getattribute(p)
    assert isinstance(all_attrs, dict)
    assert all_attrs["delta_t"] == 0.02
    assert io.getattribute(p, "delta_t") == 0.02


def test_getvar_parses_numbers_and_strings():
    s = "a=1_b=2.5_name=test_x10"
    out = io.getvar(s)
    assert out["a"] == 1
    assert out["b"] == 2.5
    assert out["name"] == "test"
    assert out["x"] == 10

    out_s = io.getvar(s, mode="strings")
    assert out_s["a"] == "1"
    assert out_s["b"] == "2.5"


def test_loadvec_bracket_expansion_and_glob(tmp_path: pathlib.Path):
    data_root = _pivpy_data_path()
    src = data_root / "openpiv_txt" / "exp1_001_b.txt"
    assert src.exists()

    p1 = tmp_path / "Run00001.txt"
    p2 = tmp_path / "Run00002.txt"
    p1.write_bytes(src.read_bytes())
    p2.write_bytes(src.read_bytes())

    out = io.loadvec(str(tmp_path / "Run[1:2].txt"))
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(d, xr.Dataset) for d in out)
    assert all("u" in d and "v" in d for d in out)


def test_loadvec_numeric_index_selects_from_cwd(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    data_root = _pivpy_data_path()
    src = data_root / "openpiv_txt" / "exp1_001_b.txt"
    assert src.exists()

    p = tmp_path / "Run00001.txt"
    p.write_bytes(src.read_bytes())

    monkeypatch.chdir(tmp_path)
    ds = io.loadvec(1)
    assert isinstance(ds, xr.Dataset)
    assert "u" in ds and "v" in ds


def test_loadarrayvec_loads_nested_lists(tmp_path: pathlib.Path):
    data_root = _pivpy_data_path()
    src = data_root / "openpiv_txt" / "exp1_001_b.txt"
    assert src.exists()

    d1 = tmp_path / "A"
    d2 = tmp_path / "B"
    d1.mkdir()
    d2.mkdir()

    for d in (d1, d2):
        (d / "Run00001.txt").write_bytes(src.read_bytes())
        (d / "Run00002.txt").write_bytes(src.read_bytes())

    out = io.loadarrayvec(str(tmp_path / "*"), "Run[1:2].txt")
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(len(row) == 2 for row in out)
    assert all(isinstance(ds, xr.Dataset) for row in out for ds in row)


def test_getfilenum_extracts_numbers(tmp_path: pathlib.Path):
    (tmp_path / "Run00001.txt").write_text("0 0 0 0\n")
    (tmp_path / "Run00002.txt").write_text("0 0 0 0\n")

    nums = io.getfilenum(str(tmp_path / "Run*.txt"), "Run", opt="fileonly")
    assert nums == [1.0, 2.0]


def test_getpivtime_uses_delta_t_and_can_shift_to_zero():
    ds = io.create_sample_Dataset(n_frames=3, rows=3, cols=3, noise_sigma=0.0)
    ds.attrs["delta_t"] = 0.1
    # Make time coord non-zero start to verify shifting behavior.
    ds = ds.assign_coords(t=("t", np.asarray([5.0, 6.0, 7.0])))

    t = io.getpivtime(ds)
    assert np.allclose(t, [0.5, 0.6, 0.7])

    t0 = io.getpivtime(ds, "0")
    assert np.allclose(t0, [0.0, 0.1, 0.2])


def test_getimx_returns_expected_shapes_for_scalar_and_vector():
    im = np.arange(12, dtype=float).reshape(3, 4)
    ds_w = io.im2pivmat(im)
    x2d, y2d, w = io.getimx(ds_w)
    assert x2d.shape == (3, 4)
    assert y2d.shape == (3, 4)
    assert w.shape == (3, 4)

    ds_uv = io.create_sample_field(rows=3, cols=4)
    x2d, y2d, u, v, chc = io.getimx(ds_uv)
    assert u.shape == (3, 4)
    assert v.shape == (3, 4)
    assert chc.shape == (3, 4)


def test_readvec_parses_header_and_shapes(tmp_path: pathlib.Path):
    p = tmp_path / "demo.vec"
    header = 'variables="X" "mm" "Y" "mm" "U" "m/s" "V" "m/s" i=2 j=3\n'
    # 6 rows, 4 columns
    body = "\n".join(
        [
            "0 0 1 10",
            "1 0 2 20",
            "0 1 3 30",
            "1 1 4 40",
            "0 2 5 50",
            "1 2 6 60",
        ]
    )
    p.write_text(header + body + "\n")

    hdr, data = io.readvec(p, comments=1)
    assert "variables" in hdr
    assert data.shape == (3, 2, 4)
    assert np.isfinite(data).all()


def test_openim_requires_lvpyio_when_missing():
    import importlib.util

    if importlib.util.find_spec("lvpyio") is not None:
        pytest.skip("lvpyio installed; openim* needs real test fixtures")

    with pytest.raises(ImportError):
        io.openim7("dummy.im7")
    with pytest.raises(ImportError):
        io.openimx("dummy.imx")
    with pytest.raises(ImportError):
        io.openimg("dummy.img")


def test_vortex_and_multivortex_basic_properties():
    ds = io.vortex(n=32, r0=6, vorticity=1.0, mode="burgers")
    assert isinstance(ds, xr.Dataset)
    assert set(["u", "v"]).issubset(ds.data_vars)
    assert ds["u"].shape == (32, 32, 1)
    assert float(np.nanmax(np.abs(ds["u"].values))) > 0.0

    ds2 = io.multivortex(nfield=3, nsize=32, numvortex=2)
    assert ds2.sizes["t"] == 3
    assert ds2["u"].shape == (32, 32, 3)

    # Deterministic seed: same call should reproduce.
    ds3 = io.multivortex(nfield=3, nsize=32, numvortex=2)
    assert np.allclose(ds2["u"].values, ds3["u"].values)


def test_specf_and_spec2f_vector_outputs_and_even_enforcement():
    ds = io.create_sample_Dataset(n_frames=2, rows=9, cols=9, noise_sigma=0.0)
    out1 = specf(ds)
    assert set(["exvx", "exvy", "eyvx", "eyvy"]).issubset(out1.data_vars)
    # odd inputs drop to even -> 8 -> nk=4
    assert out1.sizes["kx"] == 4
    assert out1.sizes["ky"] == 4
    assert "k" in out1.coords
    assert "e" in out1.data_vars

    out2 = spec2f(ds)
    assert "e" in out2.data_vars
    assert out2["e"].ndim == 2
    assert out2.sizes["kx"] == 4
    assert out2.sizes["ky"] == 4
    assert "k" in out2
    assert "ep" in out2


def test_specf_scalar_from_im2pivmat():
    im = np.random.default_rng(0).standard_normal((8, 8))
    ds = io.im2pivmat(im)
    out = specf(ds, "hann")
    assert set(["ex", "ey"]).issubset(out.data_vars)
    assert out.sizes["kx"] == 4
    assert out.sizes["ky"] == 4
    assert "k" in out.coords
    assert "e" in out.data_vars


def test_tempspecf_vector_requires_time_and_handles_zero_option():
    ds = io.create_sample_Dataset(n_frames=8, rows=3, cols=3, noise_sigma=0.0)

    # Inject a zero into every time series.
    ds = ds.copy(deep=True)
    ds["u"].loc[{"t": ds["t"].values[0]}] = 0.0

    out_nozero = tempspecf(ds, 10.0)
    assert out_nozero.sizes["w"] == 4
    assert np.allclose(out_nozero["e"].values, 0.0)

    out_zero = tempspecf(ds, 10.0, "zero")
    assert out_zero.sizes["w"] == 4
    assert float(np.nanmax(out_zero["e"].values)) > 0.0


def test_ssf_and_vsf_shapes_and_required_vars():
    ds_w = io.im2pivmat(np.random.default_rng(0).standard_normal((12, 12)))
    bins = np.linspace(-1.0, 1.0, 21)
    r = np.asarray([1, 2, 3], dtype=int)

    out_s = ssf(ds_w, "x", *["r", r, "bin", bins, "maxorder", 4])
    assert out_s.sizes["r"] == 3
    assert out_s.sizes["bin"] == 21
    assert out_s.sizes["order"] == 4
    assert "sf" in out_s and "sfabs" in out_s

    ds_uv = io.create_sample_field(rows=12, cols=12)
    out_v = vsf(ds_uv, *["r", r, "bin", bins, "maxorder", 4])
    assert out_v.sizes["r"] == 3
    assert out_v.sizes["bin"] == 21
    assert out_v.sizes["order"] == 4
    assert set(["lsf", "tsf", "n_long", "n_trans"]).issubset(out_v.data_vars)


def test_operf_unary_and_binary_vector():
    ds = io.create_sample_field(rows=3, cols=4)
    out_neg = operf("-", ds)
    assert np.allclose(out_neg["u"].values, -ds["u"].values)
    assert np.allclose(out_neg["v"].values, -ds["v"].values)

    ds2 = io.create_sample_field(rows=3, cols=4)
    out_add = operf("+", ds, ds2)
    assert np.allclose(out_add["u"].values, ds["u"].values + ds2["u"].values)
    assert np.allclose(out_add["v"].values, ds["v"].values + ds2["v"].values)


def test_operf_numeric_vector_add_and_scale():
    ds = io.create_sample_field(rows=3, cols=4)
    out_add = operf("+", ds, np.asarray([1.0, 2.0]))
    assert np.allclose(out_add["u"].values, ds["u"].values + 1.0)
    assert np.allclose(out_add["v"].values, ds["v"].values + 2.0)

    out_mul = operf("*", ds, 2.0)
    assert np.allclose(out_mul["u"].values, ds["u"].values * 2.0)
    assert np.allclose(out_mul["v"].values, ds["v"].values * 2.0)


def test_operf_threshold_and_binarize_vector():
    ds = io.create_sample_field(rows=3, cols=4)
    out_thr = operf(">", ds, 3.0)
    assert np.all((out_thr["u"].values == 0.0) | (out_thr["u"].values > 3.0))

    out_bin = operf("b>", ds, 3.0)
    assert set(np.unique(out_bin["u"].values)).issubset({0.0, 1.0})
    assert set(np.unique(out_bin["v"].values)).issubset({0.0, 1.0})


def test_operf_list_behavior_uses_single_rhs():
    a = io.create_sample_field(rows=3, cols=4)
    b = io.create_sample_field(rows=3, cols=4)
    rhs = io.create_sample_field(rows=3, cols=4)
    out = operf("+", [a, b], [rhs])
    assert isinstance(out, list)
    assert len(out) == 2
    assert np.allclose(out[0]["u"].values, a["u"].values + rhs["u"].values)
    assert np.allclose(out[1]["u"].values, b["u"].values + rhs["u"].values)


def test_randvec_shapes_determinism_and_near_zero_divergence():
    ds = io.randvec(32, 3, seed=0)
    assert isinstance(ds, xr.Dataset)
    assert set(["u", "v", "chc"]).issubset(ds.data_vars)
    assert ds["u"].shape == (32, 32, 3)
    assert ds.sizes["t"] == 3

    # Deterministic by default seed.
    ds2 = io.randvec(32, 3, seed=0)
    assert np.allclose(ds["u"].values, ds2["u"].values)
    assert np.allclose(ds["v"].values, ds2["v"].values)

    # Check that mean flow is ~0 (zero Fourier mode).
    assert abs(float(ds["u"].isel(t=0).mean())) < 1e-8
    assert abs(float(ds["v"].isel(t=0).mean())) < 1e-8

    # Divergence-free in Fourier space, excluding Nyquist modes.
    # (Nyquist lines are special for real fields; PIVMAT handles them separately.)
    n = 32
    u = np.asarray(ds["u"].isel(t=0).values, dtype=float)
    v = np.asarray(ds["v"].isel(t=0).values, dtype=float)
    U = np.fft.fft2(u) / (n**2)
    V = np.fft.fft2(v) / (n**2)

    k1d = (np.fft.fftfreq(n) * n).astype(float)
    kx2d, ky2d = np.meshgrid(k1d, k1d)
    K = np.hypot(kx2d, ky2d)
    mask = (K > 0) & (np.abs(kx2d) != n / 2) & (np.abs(ky2d) != n / 2)
    kdot = kx2d * U + ky2d * V

    num = float(np.sqrt(np.mean(np.abs(kdot[mask]) ** 2)))
    den = float(np.sqrt(np.mean((K[mask] * np.hypot(np.abs(U[mask]), np.abs(V[mask]))) ** 2)))
    assert num / (den + 1e-12) < 1e-10


def test_randvec_requires_even_n():
    with pytest.raises(ValueError):
        io.randvec(31, 1)
