import warnings

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

try:
    from scipy.ndimage import convolve as _nd_convolve
except Exception:  # pragma: no cover
    _nd_convolve = None

try:
    from scipy.signal import convolve2d as _signal_convolve2d
except Exception:  # pragma: no cover
    _signal_convolve2d = None


def _kernel_flat(filtsize: float) -> np.ndarray:
    n = int(np.ceil(float(filtsize)))
    if n <= 0:
        return np.ones((1, 1), dtype=float)
    k = np.ones((n, n), dtype=float)
    k /= float(np.sum(k))
    return k


def _kernel_gauss(filtsize: float, *, integrated: bool = False) -> np.ndarray:
    fs = float(filtsize)
    if fs <= 0.0 or not np.isfinite(fs):
        return np.ones((1, 1), dtype=float)

    half_width = int(np.ceil(3.5 * fs))
    coords = np.arange(-half_width, half_width + 1, dtype=float)

    if integrated:
        # Integrated Gaussian over pixel area [-0.5, 0.5], to reduce discretization
        # effects for small filtsize.
        try:
            from scipy.special import erf  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("filterf(method='igauss') requires SciPy (scipy.special.erf)") from exc

        s = np.sqrt(2.0) * fs
        left = (coords - 0.5) / s
        right = (coords + 0.5) / s
        g1 = 0.5 * (erf(right) - erf(left))
        k = np.outer(g1, g1)
    else:
        X, Y = np.meshgrid(coords, coords)
        k = np.exp(-(X * X + Y * Y) / (2.0 * fs * fs))

    k = np.asarray(k, dtype=float)
    s = float(np.sum(k))
    if not np.isfinite(s) or s == 0.0:
        return np.ones((1, 1), dtype=float)
    k /= s
    return k


def filter2d_kernel(filtsize: float = 1.0, method: str = "gauss") -> np.ndarray:
    """Return the normalized 2D kernel used by :func:`filter2d`.

    Parameters
    ----------
    filtsize:
        Filter size in mesh units.
    method:
        'gauss', 'flat', or 'igauss'.
    """

    m = str(method).lower()
    fs = float(filtsize)
    if m.startswith("flat"):
        return _kernel_flat(fs)
    if m.startswith("igauss"):
        return _kernel_gauss(fs, integrated=True)
    return _kernel_gauss(fs, integrated=False)


def filter2d(
    arr2: ArrayLike,
    filtsize: float = 1.0,
    method: str = "gauss",
    *,
    mode: str = "valid",
) -> np.ndarray:
    """2D spatial filter by normalized convolution (PIVMAT-inspired).

    Parameters
    ----------
    arr2:
        2D field to filter.
    filtsize:
        Filter size in mesh units (Gaussian sigma-like scale).
    method:
        'gauss', 'flat', or 'igauss' (integrated Gaussian).
    mode:
        'valid' (default) or 'same'. Matches Matlab conv2 modes.

    Notes
    -----
    Missing values are treated as NaN and excluded from the convolution by
    using a weight-mask normalization.
    """

    a = np.asarray(arr2, dtype=float)
    if a.ndim != 2:
        raise ValueError("filter2d expects a 2D array")

    fs = float(filtsize)
    if fs == 0.0:
        return a

    m = str(method).lower()
    if m.startswith("flat"):
        k = _kernel_flat(fs)
    elif m.startswith("igauss"):
        k = _kernel_gauss(fs, integrated=True)
    else:
        k = _kernel_gauss(fs, integrated=False)

    if _signal_convolve2d is None:  # pragma: no cover
        raise ImportError("filter2d requires scipy.signal.convolve2d")

    finite = np.isfinite(a)
    a0 = np.where(finite, a, 0.0)
    w = finite.astype(float)

    mode_l = str(mode).lower()
    if mode_l not in ("valid", "same"):
        raise ValueError("mode must be 'valid' or 'same'")

    num = _signal_convolve2d(a0, k, mode=mode_l, boundary="fill", fillvalue=0.0)
    den = _signal_convolve2d(w, k, mode=mode_l, boundary="fill", fillvalue=0.0)

    out = np.full_like(num, np.nan, dtype=float)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def bwfilter2d(arr2: np.ndarray, filtsize: float, order: float) -> np.ndarray:
    """2D Butterworth filter in Fourier space (PIVMAT-inspired).

    Parameters
    ----------
    arr2:
        2D array.
    filtsize:
        Cutoff size in grid units. If 0, returns input.
    order:
        Butterworth order. Positive -> low-pass. Negative -> high-pass.

    Notes
    -----
    This follows the PIVMAT convention:
    - k is measured in index space on the FFT-shifted grid.
    - kc = n / filtsize, where n = min(nx, ny).
    - Transfer: T(k)=1/(1+(k/kc)^(order/2)).
    """

    a = np.asarray(arr2, dtype=float)
    if a.ndim != 2:
        raise ValueError("bwfilter2d expects a 2D array")

    fs = float(filtsize)
    if fs == 0.0:
        return a

    ny, nx = a.shape
    # PIVMAT behavior: if odd, discard last row/col.
    if nx % 2 == 1:
        a = a[:, :-1]
        nx -= 1
    if ny % 2 == 1:
        a = a[:-1, :]
        ny -= 1

    n = float(min(nx, ny))
    kc = n / fs
    if not np.isfinite(kc) or kc == 0.0:
        return a

    # Integer-like wavenumbers on the shifted FFT grid: [-N/2 .. N/2-1]
    kx = np.fft.fftshift(np.fft.fftfreq(nx) * nx)
    ky = np.fft.fftshift(np.fft.fftfreq(ny) * ny)
    KX, KY = np.meshgrid(kx, ky)
    k = np.sqrt(KX * KX + KY * KY)

    p = float(order) / 2.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        powterm = np.power(k / kc, p)
        T = 1.0 / (1.0 + powterm)

    # Fix the zero-mode explicitly for numerical stability.
    if order < 0:
        T[k == 0] = 0.0
    else:
        T[k == 0] = 1.0

    sp = np.fft.fftshift(np.fft.fft2(a))
    out = np.fft.ifft2(np.fft.ifftshift(sp * T)).real
    return out


def corrx(
    x: np.ndarray,
    y: np.ndarray | None = None,
    *,
    half: bool = False,
    nan_as_zero: bool = True,
) -> np.ndarray:
    """Vector correlation (PIVMAT-compatible).

    This ports the behavior of PIVMAT's ``corrx.m``. Zero-padding is used
    outside the signal support, and each lag is normalized by the number of
    non-zero products (so missing data encoded as zeros does not bias the
    result).

    Parameters
    ----------
    x, y:
        1D vectors. If ``y`` is None, autocorrelation is computed.
    half:
        If True, return only non-negative lags (including zero-lag).
    nan_as_zero:
        If True, NaNs are treated as missing data and replaced by 0.

    Returns
    -------
    numpy.ndarray
        Correlation vector of length ``2*N-1`` (or ``N`` if ``half=True``).
    """

    x_arr = np.asarray(x)
    if x_arr.ndim != 1:
        x_arr = x_arr.reshape(-1)

    y_arr = x_arr if y is None else np.asarray(y)
    if y_arr.ndim != 1:
        y_arr = y_arr.reshape(-1)

    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("Vectors lengths must agree.")

    x_arr = x_arr.astype(float, copy=False)
    y_arr = y_arr.astype(float, copy=False)
    if nan_as_zero:
        x_arr = np.nan_to_num(x_arr, nan=0.0)
        y_arr = np.nan_to_num(y_arr, nan=0.0)

    n = int(x_arr.shape[0])
    if n == 0:
        return np.array([], dtype=float)

    y_pad = np.concatenate([np.zeros(n - 1, dtype=float), y_arr, np.zeros(n - 1, dtype=float)])

    c = np.zeros(2 * n - 1, dtype=float)
    # MATLAB: for i=(-n+1):(n-1)
    # Python: i in [-(n-1), ..., (n-1)]
    for i in range(-(n - 1), n):
        start = (n - i - 1)
        stop = (2 * n - i - 1)
        segment = y_pad[start:stop]
        prod = x_arr * segment
        weight = int(np.count_nonzero(prod))
        if weight == 0:
            weight = 1
        c[(n - 1) + i] = float(np.sum(prod) / weight)

    if half:
        c = c[(n - 1) :]

    return c


def corrm(
    x: xr.DataArray | np.ndarray,
    dim: int | str = 1,
    *,
    half: bool = False,
    nan_as_zero: bool = True,
    lag_dim: str = "lag",
) -> xr.DataArray | np.ndarray:
    """Matrix correlation along one dimension (PIVMAT-compatible).

    For a 2D matrix shaped (M, N):
    - ``dim=1`` correlates each column vector -> output shape (2*M-1, N)
    - ``dim=2`` correlates each row vector    -> output shape (M, 2*N-1)

    For xarray objects, ``dim`` may be a dimension name and the function
    generalizes to N-D by correlating along that dimension.
    """

    if isinstance(x, xr.DataArray):
        if isinstance(dim, int):
            if dim not in (1, 2):
                raise ValueError("For xarray inputs, dim as int must be 1 or 2.")
            if x.ndim < dim:
                raise ValueError(f"Input has only {x.ndim} dims; cannot use dim={dim}.")
            dim_name = x.dims[dim - 1]
        else:
            dim_name = dim
            if dim_name not in x.dims:
                raise ValueError(f"Dimension '{dim_name}' not found in DataArray dims {x.dims}.")

        n = int(x.sizes[dim_name])
        if n == 0:
            out = xr.full_like(x.isel({dim_name: slice(0, 0)}), np.nan)
            return out

        def _corrx_1d(vec: np.ndarray) -> np.ndarray:
            return corrx(vec, half=False, nan_as_zero=nan_as_zero)

        out = xr.apply_ufunc(
            _corrx_1d,
            x,
            input_core_dims=[[dim_name]],
            output_core_dims=[[lag_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # Put the lag dimension where the original dim was.
        dim_index = list(x.dims).index(dim_name)
        desired_order = list(x.dims)
        desired_order[dim_index] = lag_dim
        out = out.transpose(*desired_order)

        lag = np.arange(-(n - 1), n, dtype=int)
        out = out.assign_coords({lag_dim: lag})

        if half:
            out = out.sel({lag_dim: slice(0, None)})

        return out

    # NumPy path (expects 2D for PIVMAT-like behavior)
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError("NumPy corrm expects a 2D array; for N-D use xarray DataArray.")

    m, n = arr.shape
    if dim == 2:
        c = np.zeros((m, 2 * n - 1), dtype=float)
        for i in range(m):
            c[i, :] = corrx(arr[i, :], half=False, nan_as_zero=nan_as_zero)
        if half:
            c = c[:, (n - 1) :]
        return c
    if dim == 1:
        c = np.zeros((2 * m - 1, n), dtype=float)
        for j in range(n):
            c[:, j] = corrx(arr[:, j], half=False, nan_as_zero=nan_as_zero)
        if half:
            c = c[(m - 1) :, :]
        return c

    raise ValueError("dim must be 1 or 2 (NumPy inputs) or a valid xarray dim name.")


def _corrf_scales(
    r: np.ndarray,
    f: np.ndarray,
    *,
    nowarning: bool = False,
) -> dict[str, float]:
    """Compute integral scales and crossover radii for a 1D correlation curve.

    This follows the PIVMAT `corrf.m` convention:
    - isinf integrates from r=0 to r=r_max using a simple Riemann sum.
    - r0/r1/r2/r5 are the first crossover radii for thresholds 0, 0.1, 0.2, 0.5
      (linearly interpolated between samples).
    - is0/is1/is2/is5 integrate from r=0 up to the first index where the
      threshold is crossed (inclusive), normalized by f(0).
    """

    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=float)
    if r.ndim != 1 or f.ndim != 1 or r.shape[0] != f.shape[0]:
        raise ValueError("r and f must be 1D arrays of equal length")
    if r.shape[0] == 0:
        nan = float("nan")
        return {
            "isinf": nan,
            "r0": nan,
            "is0": nan,
            "r1": nan,
            "is1": nan,
            "r2": nan,
            "is2": nan,
            "r5": nan,
            "is5": nan,
        }

    f0 = float(f[0])
    if not np.isfinite(f0) or f0 == 0.0:
        nan = float("nan")
        return {
            "isinf": nan,
            "r0": nan,
            "is0": nan,
            "r1": nan,
            "is1": nan,
            "r2": nan,
            "is2": nan,
            "r5": nan,
            "is5": nan,
        }

    if r.shape[0] >= 2:
        dr = float(r[1] - r[0])
    else:
        dr = 1.0

    def _first_crossing(target: float, *, use_normalized: bool) -> tuple[float, float]:
        if use_normalized:
            series = f / f0
            thr = target
        else:
            series = f
            thr = target

        idx = np.where(series <= thr)[0]
        if idx.size == 0:
            if not nowarning:
                warnings.warn(
                    f"Correlation function does not cross threshold {target}.",
                    UserWarning,
                    stacklevel=2,
                )
            return float("nan"), float("nan")

        k = int(idx[0])
        if k == 0:
            return float(r[0]), 0.0

        # Linear interpolation (PIVMAT-style): use samples k-1 and k.
        r0_ = float(r[k])
        r1_ = float(r[k - 1])
        f0_ = float(series[k])
        f1_ = float(series[k - 1])

        denom = (f0_ - f1_)
        if denom == 0.0:
            rt = r0_
        else:
            rt = r0_ + (thr - f0_) * ((r0_ - r1_) / denom)

        # Integral scale: sum up to index k (inclusive), normalized.
        is_t = float(np.sum((f[: (k + 1)] / f0)) * dr)
        return float(rt), float(is_t)

    isinf = float(np.sum((f / f0)) * dr)
    r0, is0 = _first_crossing(0.0, use_normalized=False)
    r1, is1 = _first_crossing(0.1, use_normalized=True)
    r2, is2 = _first_crossing(0.2, use_normalized=True)
    r5, is5 = _first_crossing(0.5, use_normalized=True)

    return {
        "isinf": isinf,
        "r0": r0,
        "is0": is0,
        "r1": r1,
        "is1": is1,
        "r2": r2,
        "is2": is2,
        "r5": r5,
        "is5": is5,
    }


def corrf(
    x: xr.DataArray,
    dim: int | str = "x",
    *,
    normalize: bool = False,
    nan_as_zero: bool = True,
    nowarning: bool = False,
    r_dim: str = "r",
) -> xr.Dataset:
    """Spatial correlation function and integral scales (PIVMAT-inspired).

    This is a Python/xarray equivalent of PIVMAT's ``corrf.m`` for a scalar field.

    The correlation along a direction is defined (conceptually) as:
    f(r) = < F(x,y) F(x+r,y) >
    where <..> denotes spatial averaging over the orthogonal
    direction and ensemble averaging over any remaining dimensions (e.g. time).

    Parameters
    ----------
    x:
        Scalar field as an ``xarray.DataArray`` (typically with dims including
        ``'x'`` and ``'y'``, and optionally ``'t'``).
    dim:
        Direction of separation: ``'x'``/``'y'`` (recommended) or MATLAB-like
        ``1``/``2``.
    normalize:
        If True, normalize the correlation so that ``f(0)=1``.
    nan_as_zero:
        If True, treat NaNs as missing data and replace by 0 before correlating.
        (Missing values encoded as 0 are handled by the PIVMAT-style weighting in
        ``corrx``/``corrm``.)
    nowarning:
        If True, suppress warnings when crossover radii are undefined.
    r_dim:
        Name of the separation-length coordinate in the returned Dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with 1D variable ``f`` over coordinate ``r`` and scalar
        variables ``isinf, r0, is0, r1, is1, r2, is2, r5, is5``.
    """

    if not isinstance(x, xr.DataArray):
        raise TypeError("corrf expects an xarray.DataArray")

    # Resolve the dimension name for coordinate spacing.
    if isinstance(dim, str):
        dim_name = dim
        if dim_name in ("x", "y") and dim_name not in x.dims:
            # Allow 'x'/'y' mapping only if present; otherwise let corrm raise.
            pass
    else:
        if dim not in (1, 2):
            raise ValueError("dim must be 'x', 'y', 1 or 2")
        if x.ndim < dim:
            raise ValueError(f"Input has only {x.ndim} dims; cannot use dim={dim}.")
        dim_name = x.dims[dim - 1]

    c = corrm(x, dim=dim, half=True, nan_as_zero=nan_as_zero, lag_dim="lag")

    # Average over all dimensions except lag.
    mean_dims = [d for d in c.dims if d != "lag"]
    f_da = c.mean(dim=mean_dims, skipna=True)

    # Separation length: lag index (0..N-1) times grid spacing.
    if dim_name in x.coords and x.sizes.get(dim_name, 0) >= 2:
        coord = x[dim_name].values
        diffs = np.diff(coord.astype(float))
        dr = float(np.abs(diffs[0])) if diffs.size else 1.0
        if diffs.size and not np.allclose(diffs, diffs[0]):
            if not nowarning:
                warnings.warn(
                    f"Non-uniform spacing detected along '{dim_name}'; using first step for dr.",
                    UserWarning,
                    stacklevel=2,
                )
    else:
        dr = 1.0

    lag = f_da["lag"].values.astype(float)
    r = lag * dr
    f = f_da.values.astype(float)

    if normalize and f.size:
        if f[0] != 0.0:
            f = f / float(f[0])

    scales = _corrf_scales(r, f, nowarning=nowarning)

    out = xr.Dataset(coords={r_dim: r})
    out["f"] = (r_dim, f)
    for k, v in scales.items():
        out[k] = xr.DataArray(v)

    out.attrs["dim"] = str(dim)
    out.attrs["dr"] = float(dr)
    out.attrs["normalized"] = bool(normalize)
    out.attrs["variable"] = str(x.name) if x.name is not None else ""

    return out


def gradientf(scalar: xr.DataArray) -> xr.Dataset:
    """Gradient of a scalar field (PIVMAT-inspired).

    This is a Python/xarray equivalent of PIVMAT's ``gradientf.m``.

    Parameters
    ----------
    scalar:
        Scalar field as an ``xarray.DataArray`` with dims including ``x`` and ``y``.
        A time dimension (typically ``t``) is allowed and is preserved.

    Returns
    -------
    xarray.Dataset
        Dataset with variables ``u`` and ``v`` containing the partial derivatives
        ``d(scalar)/dx`` and ``d(scalar)/dy``.

    Notes
    -----
    - Coordinates are taken from the input DataArray.
    - Units are propagated if both the scalar and coordinate units are available.
    """

    if not isinstance(scalar, xr.DataArray):
        raise TypeError("gradientf expects an xarray.DataArray")

    if "x" not in scalar.dims or "y" not in scalar.dims:
        raise ValueError("gradientf requires dims 'x' and 'y'")

    gx = scalar.differentiate("x")
    gy = scalar.differentiate("y")

    out = xr.Dataset({"u": gx, "v": gy})

    # Best-effort metadata propagation.
    w_units = str(scalar.attrs.get("units", ""))
    x_units = str(getattr(scalar.coords.get("x", None), "attrs", {}).get("units", ""))
    y_units = str(getattr(scalar.coords.get("y", None), "attrs", {}).get("units", ""))

    name = scalar.name or "scalar"
    out["u"].attrs = dict(gx.attrs)
    out["v"].attrs = dict(gy.attrs)
    out["u"].attrs.setdefault("long_name", f"d/dx({name})")
    out["v"].attrs.setdefault("long_name", f"d/dy({name})")

    if w_units and x_units:
        out["u"].attrs["units"] = f"{w_units}/{x_units}"
    if w_units and y_units:
        out["v"].attrs["units"] = f"{w_units}/{y_units}"

    return out


def _hist_counts(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Histogram counts using bin centers (MATLAB hist-like behavior)."""

    c = np.asarray(centers, dtype=float).ravel()
    if c.size == 0:
        return np.asarray([], dtype=int)

    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros(c.size, dtype=int)

    # Use midpoints between centers as bin edges, with infinite end caps.
    mids = 0.5 * (c[:-1] + c[1:])
    edges = np.concatenate(([-np.inf], mids, [np.inf]))
    h, _ = np.histogram(v, bins=edges)
    return np.asarray(h, dtype=int)


def histf(
    scalar: xr.DataArray,
    bin: np.ndarray | None = None,
    opt: str = "",
) -> xr.Dataset:
    """Histogram of a scalar field (PIVMAT-inspired).

    This is a Python/xarray equivalent of PIVMAT's ``histf.m`` for scalar
    fields. Values are stacked over all dimensions.

    Parameters
    ----------
    scalar:
        Scalar field as an ``xarray.DataArray``.
    bin:
        Optional 1D array of bin centers. If not provided, a default binning is
        estimated from the mean and standard deviation of the first frame
        (if a ``t`` dimension exists) or from the full field otherwise.
    opt:
        Option string. If it contains ``'0'``, zero values are included.

    Returns
    -------
    xarray.Dataset
        Dataset with coordinate ``bin`` and variable ``h``.
    """

    if not isinstance(scalar, xr.DataArray):
        raise TypeError("histf expects an xarray.DataArray")

    include_zeros = "0" in str(opt)

    da0 = scalar
    if "t" in scalar.dims:
        try:
            da0 = scalar.isel(t=0)
        except Exception:
            da0 = scalar

    ref = np.asarray(da0.values, dtype=float).ravel()
    ref = ref[np.isfinite(ref)]
    if not include_zeros:
        ref = ref[ref != 0]

    if bin is None:
        if ref.size == 0:
            centers = np.linspace(-1.0, 1.0, 200)
        else:
            mean = float(np.mean(ref))
            std = float(np.std(ref))
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            if mean < std:
                centers = np.linspace(-20.0 * std, 20.0 * std, 200)
            else:
                centers = np.linspace(mean - 20.0 * std, mean + 20.0 * std, 200)
    else:
        centers = np.asarray(bin, dtype=float).ravel()

    vals = np.asarray(scalar.values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if not include_zeros:
        vals = vals[vals != 0]

    h = _hist_counts(vals, centers)
    out = xr.Dataset({"h": ("bin", h)}, coords={"bin": centers})
    out["h"].attrs["long_name"] = "histogram"
    return out


def meannz(
    x: xr.DataArray | np.ndarray,
    dim: int | str | None = None,
    *,
    keep_attrs: bool = True,
) -> xr.DataArray | np.ndarray:
    """Mean of nonzero elements (PIVMAT-compatible).

    Matches the intent of PIVMAT's ``meannz.m``: normalize by the number of
    nonzero samples instead of the total number of samples.

    Notes
    -----
    - For xarray inputs, zeros are excluded but NaNs are skipped.
    - Where a reduction slice has no nonzero samples, the result is 0.
    """

    if isinstance(x, xr.DataArray):
        if dim is None:
            # First non-singleton dimension (Matlab-like).
            dim_name = next((d for d in x.dims if x.sizes[d] != 1), x.dims[0])
        elif isinstance(dim, int):
            dim_name = x.dims[int(dim)]
        else:
            dim_name = dim

        nz = x.where(x != 0)
        summed = nz.sum(dim=dim_name, skipna=True)
        count = (x != 0).sum(dim=dim_name)
        out = summed / count.where(count != 0)
        out = out.fillna(0.0)
        if keep_attrs:
            out.attrs = dict(x.attrs)
        return out

    arr = np.asarray(x)
    if dim is None:
        dim = next((i for i, s in enumerate(arr.shape) if s != 1), 0)
    if not isinstance(dim, int):
        raise ValueError("For NumPy inputs, dim must be an int or None.")

    arr_f = arr.astype(float, copy=False)
    mask = arr_f != 0
    summed = np.sum(np.where(mask, arr_f, 0.0), axis=dim)
    count = np.sum(mask, axis=dim)
    out = np.divide(summed, count, out=np.zeros_like(summed, dtype=float), where=(count != 0))
    out = np.nan_to_num(out, nan=0.0)
    return out


def interpolat_zeros_2d(
    m: xr.DataArray | np.ndarray,
    *,
    fill: bool = False,
    max_iter: int | None = None,
    nan_as_zero: bool = True,
) -> xr.DataArray | np.ndarray:
    """Interpolate zeros in a 2D field using 4-neighbor averaging.

    This mirrors the behavior of PIVMAT's legacy ``interpolat.m``:
    - Replaces zero entries by the mean of their nonzero 4-neighbors.
    - If ``fill=True``, repeats until no zeros remain (or ``max_iter``).

    Parameters
    ----------
    m:
        2D array (or xarray DataArray with at least two dims).
    fill:
        Iterate until no zeros remain.
    max_iter:
        Optional hard stop for iterations (recommended when fill=True).
    nan_as_zero:
        If True, NaNs are treated as 0 (missing/invalid).
    """

    def _interp_pass(a2: np.ndarray) -> np.ndarray:
        if _nd_convolve is None:
            raise ImportError("scipy is required for interpolat_zeros_2d")

        a2 = a2.astype(float, copy=True)
        if nan_as_zero:
            a2 = np.nan_to_num(a2, nan=0.0)

        zero_mask = a2 == 0
        if not np.any(zero_mask):
            return a2

        kernel = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        nonzero = (~zero_mask).astype(float)
        neigh_sum = _nd_convolve(a2, kernel, mode="nearest")
        neigh_cnt = _nd_convolve(nonzero, kernel, mode="nearest")

        # Only update zeros where at least one nonzero neighbor exists.
        update = zero_mask & (neigh_cnt > 0)
        a2[update] = neigh_sum[update] / neigh_cnt[update]
        return a2

    if isinstance(m, xr.DataArray):
        if m.ndim < 2:
            raise ValueError("interpolat_zeros_2d requires at least 2D input")
        y_dim, x_dim = m.dims[0], m.dims[1]

        def _core(arr2: np.ndarray) -> np.ndarray:
            out = arr2
            it = 0
            while True:
                new = _interp_pass(out)
                it += 1
                if not fill:
                    return new
                if not np.any(new == 0):
                    return new
                if max_iter is not None and it >= int(max_iter):
                    return new
                if np.array_equal(new, out):
                    return new
                out = new

        out = xr.apply_ufunc(
            _core,
            m,
            input_core_dims=[[y_dim, x_dim]],
            output_core_dims=[[y_dim, x_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        out = out.assign_coords({y_dim: m[y_dim], x_dim: m[x_dim]})
        out.attrs = dict(m.attrs)
        return out

    arr = np.asarray(m)
    if arr.ndim != 2:
        raise ValueError("interpolat_zeros_2d expects a 2D NumPy array")
    out = arr.astype(float, copy=True)
    it = 0
    while True:
        new = _interp_pass(out)
        it += 1
        if not fill:
            return new
        if not np.any(new == 0):
            return new
        if max_iter is not None and it >= int(max_iter):
            return new
        if np.array_equal(new, out):
            return new
        out = new


def inpaint_missing_2d(
    a2: ArrayLike,
    *,
    method: int = 0,
    missing: str = "0nan",
) -> np.ndarray:
    """Inpaint missing values in a 2D array (PIVMAT ``interpf``-style).

    Missing values are defined as NaNs and/or zeros.

    Parameters
    ----------
    a2:
        2D array.
    method:
        Integer method selector (PIVMAT-inspired):

        - ``0``: Laplacian (harmonic) inpainting via sparse linear solve.
        - ``1``: Nearest-neighbor fill (fast, robust).
        - ``2``: Linear interpolation via ``scipy.interpolate.griddata``.
    missing:
        Missing-value definition:
        - ``"0nan"`` (default): treat both ``0`` and ``NaN`` as missing.
        - ``"nan"``: treat only NaNs as missing.
        - ``"0"``: treat only zeros as missing.

    Returns
    -------
    numpy.ndarray
        Filled array (float).
    """

    a = np.asarray(a2, dtype=float)
    if a.ndim != 2:
        raise ValueError("inpaint_missing_2d expects a 2D array")

    missing_l = str(missing).lower()
    if missing_l not in {"0nan", "nan", "0"}:
        raise ValueError("missing must be one of: '0nan', 'nan', '0'")

    mask_nan = ~np.isfinite(a)
    mask_zero = a == 0.0
    if missing_l == "0nan":
        miss = mask_nan | mask_zero
    elif missing_l == "nan":
        miss = mask_nan
    else:
        miss = mask_zero

    if not np.any(miss):
        return a

    # If everything is missing, return zeros to match common PIV conventions.
    if np.all(miss):
        return np.zeros_like(a, dtype=float)

    m = int(method)
    if m == 1:
        # Nearest-neighbor fill via distance transform.
        try:
            from scipy.ndimage import distance_transform_edt  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("method=1 requires SciPy (scipy.ndimage.distance_transform_edt)") from exc

        valid = ~miss
        # distance_transform_edt expects False for features; compute indices of nearest valid.
        _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
        out = a.copy()
        out[miss] = out[iy[miss], ix[miss]]
        out[~np.isfinite(out)] = 0.0
        return out

    if m == 2:
        try:
            from scipy.interpolate import griddata  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("method=2 requires SciPy (scipy.interpolate.griddata)") from exc

        ny, nx = a.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        pts = np.column_stack([yy[~miss].ravel(), xx[~miss].ravel()])
        vals = a[~miss].ravel()
        xi = (yy[miss], xx[miss])
        out = a.copy()
        filled = griddata(pts, vals, xi, method="linear")
        # griddata returns NaN outside convex hull; fall back to nearest for those.
        if np.any(~np.isfinite(filled)):
            filled2 = griddata(pts, vals, xi, method="nearest")
            filled = np.where(np.isfinite(filled), filled, filled2)
        out[miss] = filled
        out[~np.isfinite(out)] = 0.0
        return out

    if m != 0:
        raise ValueError("Unsupported method. Supported: 0, 1, 2")

    # Method 0: solve Laplace equation on missing nodes with Dirichlet boundary on known nodes.
    try:
        from scipy.sparse import lil_matrix  # type: ignore
        from scipy.sparse.linalg import spsolve  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("method=0 requires SciPy sparse (scipy.sparse, scipy.sparse.linalg)") from exc

    ny, nx = a.shape
    idx = -np.ones((ny, nx), dtype=int)
    unknown_positions = np.argwhere(miss)
    n_unknown = int(unknown_positions.shape[0])
    for k, (iy, ix) in enumerate(unknown_positions):
        idx[iy, ix] = k

    A = lil_matrix((n_unknown, n_unknown), dtype=float)
    b = np.zeros(n_unknown, dtype=float)

    # 4-neighbor Laplacian stencil; adjust at borders.
    for k, (iy, ix) in enumerate(unknown_positions):
        coeff_center = 0.0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            y2 = iy + dy
            x2 = ix + dx
            if y2 < 0 or y2 >= ny or x2 < 0 or x2 >= nx:
                continue
            coeff_center += 1.0
            if miss[y2, x2]:
                A[k, idx[y2, x2]] = -1.0
            else:
                b[k] += float(a[y2, x2])
        A[k, k] = float(coeff_center)

    sol = spsolve(A.tocsr(), b)
    out = a.copy()
    out[miss] = sol
    out[~np.isfinite(out)] = 0.0
    return out


def interpf(
    data: xr.Dataset,
    *,
    method: int = 0,
    variables: list[str] | None = None,
    missing: str = "0nan",
) -> xr.Dataset:
    """Interpolate missing data in a Dataset (PIVMAT ``interpf`` port).

    Missing data are values equal to 0 and/or NaN (configurable via ``missing``).
    The interpolation is applied frame-by-frame along ``t`` if present.

    Parameters
    ----------
    data:
        Input Dataset.
    method:
        See :func:`inpaint_missing_2d`.
    variables:
        Variables to process. Default: ['u','v'] if present, else ['w'] if present,
        else all data variables.
    missing:
        See :func:`inpaint_missing_2d`.

    Returns
    -------
    xarray.Dataset
        New Dataset with missing values filled.
    """

    ds = data
    if variables is None:
        if "u" in ds.data_vars and "v" in ds.data_vars:
            variables = ["u", "v"]
        elif "w" in ds.data_vars:
            variables = ["w"]
        else:
            variables = list(ds.data_vars)

    out = ds.copy(deep=True)
    for name in variables:
        if name not in out.data_vars:
            raise KeyError(f"Variable {name} not found in dataset")
        da = out[name]
        if da.ndim < 2:
            continue

        # Determine the 2D core dims (y,x) and keep remaining dims vectorized.
        y_dim, x_dim = da.dims[0], da.dims[1]

        def _core(arr2: np.ndarray) -> np.ndarray:
            return inpaint_missing_2d(arr2, method=method, missing=missing)

        filled = xr.apply_ufunc(
            _core,
            da,
            input_core_dims=[[y_dim, x_dim]],
            output_core_dims=[[y_dim, x_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # apply_ufunc may reorder non-core dims; restore original dim order.
        try:
            filled = filled.transpose(*da.dims)
        except Exception:
            pass
        filled = filled.assign_coords({y_dim: da[y_dim], x_dim: da[x_dim]})
        filled.attrs = dict(ds[name].attrs)
        out[name] = filled

    out.attrs = dict(ds.attrs)
    return out


def jpdfscal(
    s1: xr.DataArray,
    s2: xr.DataArray,
    *,
    nbin: int = 101,
) -> xr.Dataset:
    """Joint histogram ("joint PDF" in PIVMAT terminology) of two scalar fields.

    This ports the behavior of PIVMAT's ``jpdfscal``. The output is a 2D count
    matrix over symmetric bin centers spanning ``[-max(abs(s)), +max(abs(s))]``
    for each scalar.

    Parameters
    ----------
    s1, s2:
        Scalar fields as DataArrays. They must be broadcastable to the same
        shape; non-finite pairs are ignored.
    nbin:
        Number of bin centers per axis (default: 101).

    Returns
    -------
    xarray.Dataset
        Dataset with coordinates ``bin1`` and ``bin2`` and a 2D variable ``hi``
        containing counts.
    """

    if not isinstance(s1, xr.DataArray) or not isinstance(s2, xr.DataArray):
        raise TypeError("jpdfscal expects two xarray.DataArray inputs")

    n = int(nbin)
    if n < 3 or n % 2 == 0:
        # PIVMAT uses odd default (101). Odd makes the center bin land at 0.
        raise ValueError("nbin must be an odd integer >= 3")

    a1, a2 = xr.align(s1, s2, join="exact")
    v1 = np.asarray(a1.values, dtype=float).ravel()
    v2 = np.asarray(a2.values, dtype=float).ravel()

    finite = np.isfinite(v1) & np.isfinite(v2)
    v1 = v1[finite]
    v2 = v2[finite]

    if v1.size == 0:
        max1 = 0.0
    else:
        max1 = float(np.max(np.abs(v1)))
    if v2.size == 0:
        max2 = 0.0
    else:
        max2 = float(np.max(np.abs(v2)))

    bin1 = np.linspace(-max1, max1, n, dtype=float) if max1 > 0 else np.linspace(-1.0, 1.0, n, dtype=float)
    bin2 = np.linspace(-max2, max2, n, dtype=float) if max2 > 0 else np.linspace(-1.0, 1.0, n, dtype=float)

    rg = (n - 1) / 2.0

    def _to_index(v: np.ndarray, vmax: float) -> np.ndarray:
        if vmax <= 0.0 or not np.isfinite(vmax):
            return np.full_like(v, int(rg), dtype=int)
        idx = np.rint(rg * (1.0 + (v / vmax))).astype(int)
        return np.clip(idx, 0, n - 1)

    i1 = _to_index(v1, max1)
    i2 = _to_index(v2, max2)

    hi = np.zeros((n, n), dtype=float)
    # Vectorized 2D bincount
    flat = i1 * n + i2
    bc = np.bincount(flat, minlength=n * n)
    hi[:, :] = bc.reshape((n, n))

    ds = xr.Dataset(
        data_vars={"hi": (("bin1", "bin2"), hi)},
        coords={"bin1": ("bin1", bin1), "bin2": ("bin2", bin2)},
    )

    ds["hi"].attrs["long_name"] = "joint histogram"
    # Carry basic metadata if present.
    ds.attrs["namew1"] = str(a1.attrs.get("long_name", a1.name or "s1"))
    ds.attrs["namew2"] = str(a2.attrs.get("long_name", a2.name or "s2"))
    ds.attrs["unitw1"] = str(a1.attrs.get("units", ""))
    ds.attrs["unitw2"] = str(a2.attrs.get("units", ""))
    return ds
def Γ1_moving_window_function(
        fWin: xr.Dataset,
        n: int,
) -> xr.DataArray:
    """
    This is the implementation of Γ1 function given by equation 9 in L.Graftieaux, M. Michard,
    N. Grosjean, "Combining PIV, POD and vortex identification algorithms for the study of
    unsteady turbulent swirling flows", Meas.Sci.Technol., 12(2001), p.1422-1429.
    Γ1 function is used to identify the locations of the centers of the vortices (which are
    given by the Γ1 peak values within the velocity field).
    IMPORTANT NOTICE: even though this function, theoretically, can be used on its own,  
    it is not supposed to. It is designed to be used with Dask for big
    PIV datasets. The recomendation is not to use this function on its own, but rather use
    Γ1 attribute of piv class of PIVPY package (example of usage in this case would be: ds.piv.Γ1())
    This function accepts a (2*n+1)x(2*n+1) neighborhood of one velocity vector from the
    entire velocity field in the form of Xarray dataset. And for this neighborhood only, 
    it calculates the value of Γ1.
    Also, note this function is designed in a way that assumes point P (see the referenced
    article) to coincide with the center for fWin.
    This function works only for 2D velocity field.

    Args:
        fWin (xarray.Dataset): A moving-window view of the dataset.
        n (int): Rolling window radius. The window size is ``(2*n+1) x (2*n+1)``.

    Returns:
        xarray.DataArray: Γ1 value for the given rolling window.
    """
    # We must convert fWin to numpy, because when this function was originally implemented 
    # with fWin being an xr.Dataset, it was unbelievably slow! Conversion of fWin to numpy 
    # proved to give an incredible boost in speed.
    # To speed up the things even more I put everything in one line, which is unreadable.
    # Thus, to understand, what is going one, I'm giving a break up of the line
    # (the names of the variables are taken from the referenced article):
    # PMx = fWin['xCoordinates'].to_numpy() - float(fWin['xCoordinates'][n,n])
    # PMy = fWin['yCoordinates'].to_numpy() - float(fWin['yCoordinates'][n,n])
    # PM = np.sqrt(np.add(np.square(PMx), np.square(PMy)))
    # u = fWin['u'].to_numpy()
    # v = fWin['v'].to_numpy()
    # U = (u**2 + v**2)**(0.5)
    # The external tensor product (PM ^ U) * z (see the referenced article) can be simplified as a
    # cross product for the case of the 2D velocity field: (PM x U). According to the rules of
    # cross product, for the 2D velocity field, we have (PM x U) = PM_x * v - PM_y * u. In other
    # words, we don't have to compute the sin given in the referenced article. But I am, still,
    # going to use sin down below as the variable to be consistent with the expression given
    # in the referenced article.
    # sinΘM_Γ1 = (PMx*v - PMy*u) / PM / U 
    # Γ1 = np.nansum(sinΘM_Γ1) / (((2*n+1)**2))
    # And now here goes my one-liner. Note, that I didn't put PMx, PMy, u and v calculations
    # into my line. That's because I figured out emperically that would slow down the calculations.
    # n always points to the central interrogation window (just think of it). It gives me point P.
    # Robustly extract the central-point coordinates.
    # Newer xarray versions may reorder dimensions; avoid assuming positional [n, n]
    # returns a scalar.
    xcoords = fWin['xCoordinates']
    ycoords = fWin['yCoordinates']

    def _center_scalar(arr: xr.DataArray) -> float:
        if 'rollWx' in arr.dims and 'rollWy' in arr.dims:
            sel = arr.isel(rollWx=n, rollWy=n)
        elif len(arr.dims) >= 2:
            sel = arr.isel({arr.dims[0]: n, arr.dims[1]: n})
        else:
            sel = arr
        # Reduce any remaining dims (e.g. rollWt) to a scalar.
        for d in list(sel.dims):
            sel = sel.isel({d: 0})
        return float(np.asarray(sel.values).reshape(-1)[0])

    cx = _center_scalar(xcoords)
    cy = _center_scalar(ycoords)

    PMx = np.subtract(xcoords.to_numpy(), cx)
    PMy = np.subtract(ycoords.to_numpy(), cy)    
    u = fWin['u'].to_numpy()
    v = fWin['v'].to_numpy()  
    # Since for the case when point M coincides with point P we have a 0/0 situation, we'll
    # recive a warning. To temporarily suspend that warning do the following (credit goes to
    # https://stackoverflow.com/a/29950752/10073233):
    with np.errstate(divide='ignore', invalid='ignore'):
        Γ1 = np.nanmean(np.divide(np.subtract(np.multiply(PMx,v), np.multiply(PMy,u)), np.multiply(np.sqrt(np.add(np.square(PMx), np.square(PMy))), np.sqrt(np.add(np.square(u), np.square(v))))))

    return xr.DataArray(Γ1).fillna(0.0) # fillna(0) is necessary for plotting


def Γ2_moving_window_function(
        fWin: xr.Dataset,
        n: int,
) -> xr.DataArray:
    """
    This is the implementation of Γ2 function given by equation 11 in L.Graftieaux, M. Michard,
    N. Grosjean, "Combining PIV, POD and vortex identification algorithms for the study of
    unsteady turbulent swirling flows", Meas.Sci.Technol., 12(2001), p.1422-1429.
    Γ2 function is used to identify the boundaries of the vortices in a velocity field.
    IMPORTANT NOTICE: even though this function, theoretically, can be used on its own, 
    it is not supposed to. It is designed to be used with Dask for big
    PIV datasets. The recomendation is not to use this function on its own, but rather use
    Γ2 attribute of piv class of PIVPY package (example of usage in this case would be: ds.piv.Γ2())
    This function accepts a (2*n+1)x(2*n+1) neighborhood of one velocity vector from the
    entire velocity field in the form of Xarray dataset. And for this neighborhood only, 
    it calculates the value of Γ2.
    Also, note this function is designed in a way that assumes point P (see the referenced
    article) to coincide with the center for fWin.
    And finally, the choice of convective velocity (see the referenced article) is made in the
    article: it is the average velocity within fWin.
    This function works only for 2D velocity field.

    Args:
        fWin (xarray.Dataset): A moving-window view of the dataset.
        n (int): Rolling window radius. The window size is ``(2*n+1) x (2*n+1)``.

    Returns:
        xarray.DataArray: Γ2 value for the given rolling window.
    """
    # We must convert fWin to numpy, because when this function was originally implemented 
    # with fWin being an xr.Dataset, it was unbelievably slow! Conversion of fWin to numpy 
    # proved to give an incredible boost in speed.
    # To speed up the things even more I put everything in one line, which is unreadable.
    # Thus, to understand, what is going one, I'm giving a break up of the line
    # (the names of the variables are taken from the referenced article):
    # PMx = fWin['xCoordinates'].to_numpy() - float(fWin['xCoordinates'][n,n])
    # PMy = fWin['yCoordinates'].to_numpy() - float(fWin['yCoordinates'][n,n])
    # PM = np.sqrt(np.add(np.square(PMx), np.square(PMy)))
    # u = fWin['u'].to_numpy()
    # v = fWin['v'].to_numpy()
    # We are going to include point P into the calculations of velocity UP_tilde
    # uP_tilde = np.nanmean(u)
    # vP_tilde = np.nanmean(v)
    # uDif = u - uP_tilde
    # vDif = v - vP_tilde
    # UDif = (uDif**2 + vDif**2)**(0.5)
    # The external tensor product (PM ^ UDif) * z (see the referenced article) can be simplified as a
    # cross product for the case of the 2D velocity field: (PM x UDif). According to the rules of
    # cross product, for the 2D velocity field, we have (PM x UDif) = PM_x * vDif - PM_y * uDif. 
    # I am going to use sin down below as the variable to be consistent with the expression given
    # for Γ1 function.
    # sinΘM_Γ2 = (PMx*vDif - PMy*uDif) / PM / UDif 
    # Γ2 = np.nansum(sinΘM_Γ2) / (((2*n+1)**2))
    # And now here goes my one-liner. Note, that I didn't put PMx, PMy, u and v calculations
    # into my line. That's because I figured out emperically that would slow down the calculations.
    # n always points to the central interrogation window (just think of it). It gives me point P.
    xcoords = fWin['xCoordinates']
    ycoords = fWin['yCoordinates']

    def _center_scalar(arr: xr.DataArray) -> float:
        if 'rollWx' in arr.dims and 'rollWy' in arr.dims:
            sel = arr.isel(rollWx=n, rollWy=n)
        elif len(arr.dims) >= 2:
            sel = arr.isel({arr.dims[0]: n, arr.dims[1]: n})
        else:
            sel = arr
        for d in list(sel.dims):
            sel = sel.isel({d: 0})
        return float(np.asarray(sel.values).reshape(-1)[0])

    cx = _center_scalar(xcoords)
    cy = _center_scalar(ycoords)

    PMx = np.subtract(xcoords.to_numpy(), cx)
    PMy = np.subtract(ycoords.to_numpy(), cy)    
    u = fWin['u'].to_numpy()
    v = fWin['v'].to_numpy()  
    uDif = u - np.nanmean(u)
    vDif = v - np.nanmean(v)
    # Since for the case when point M coincides with point P, we have a 0/0 situation and we'll
    # recive a warning. To temporarily suspend that warning do the following (credit goes to
    # https://stackoverflow.com/a/29950752/10073233):
    with np.errstate(divide='ignore', invalid='ignore'):
        Γ2 = np.nanmean(np.divide(np.subtract(np.multiply(PMx,vDif), np.multiply(PMy,uDif)), np.multiply(np.sqrt(np.add(np.square(PMx), np.square(PMy))), np.sqrt(np.add(np.square(uDif), np.square(vDif))))))

    return xr.DataArray(Γ2).fillna(0.0) # fillna(0) is necessary for plotting