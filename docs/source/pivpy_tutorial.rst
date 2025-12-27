=================
PIVPy tutorial
=================

This tutorial is a practical, code-first walkthrough of common PIVPy workflows.
It focuses on the xarray-based API and several PIVMAT-inspired helpers.

Prerequisites
=============

Install pivpy
-------------

Recommended (fast, reproducible):

.. code-block:: bash

   uv venv
   uv pip install pivpy

Or with optional dependencies:

.. code-block:: bash

   uv pip install 'pivpy[full]'

Load a dataset
==============

PIVPy's main object is an ``xarray.Dataset``. Most post-processing is exposed via
the accessor ``ds.piv``.

.. code-block:: python

   import pivpy.pivpy  # registers the .piv accessor
   from pivpy import io

   ds = io.read_piv('your_file.vec')

Typical dataset structure:

- Dims: ``('y', 'x', 't')``
- Variables: ``u``, ``v``, and (often) ``chc``

Create a scalar dataset from an image (PIVMAT-style ``im2pivmat``)
========================================================================================

If you have a 2D image array (shape ``(y, x)``), you can wrap it into a PIVPy
scalar Dataset using :func:`pivpy.io.im2pivmat`:

.. code-block:: python

   import numpy as np
   from pivpy import io
   import pivpy.pivpy  # registers the .piv accessor

   im = np.random.rand(128, 256)
   ds_im = io.im2pivmat(im, namew='I', unit='au')

   # Plot the scalar field
   ds_im.piv.showscal(property='w')

Save animations (PIVMAT-style ``showf`` / ``imvectomovie``)
===========================================================

PIVPy can render a Dataset as a movie efficiently by creating the Matplotlib
artists once and updating them frame-by-frame (fast renderer).

In-memory Dataset (uses the ``t`` dimension as frames):

.. code-block:: python

   import pivpy.pivpy  # registers the .piv accessor
   from pivpy import io

   ds = io.create_sample_Dataset(n_frames=20)

   # Write a movie (requires FFmpeg for mp4/avi)
   ds.piv.to_movie('movie.mp4', fps=10, show='vector', nthArr=2, scalingFactor=10.0)

   # Or collect frames as RGBA arrays (no external encoders required)
   frames = ds.piv.to_movie(None, return_frames=True, show='vector')

Streaming from disk (loads files one-by-one, PIVMAT ``imvectomovie``-style):

.. code-block:: python

   from pivpy import graphics

   # Convert many files to a movie without loading all of them into RAM
   graphics.imvectomovie('data/*.vc7', 'movie.mp4', fps=10, show='vector', nthArr=3)

Notes:

- Writing ``.mp4``/``.avi`` uses Matplotlib's FFmpeg writer (FFmpeg must be installed).
- Writing ``.gif`` uses Matplotlib's Pillow writer (Pillow must be installed).
- Use ``show='scalar'`` and ``scalar='w'`` to render scalar-only datasets.

Gradient of a scalar field (PIVMAT-style ``gradientf``)
=======================================================

Compute the gradient of any scalar variable (for example a vorticity field
stored in ``ds['w']``). The result is a new Dataset with vector components
stored as ``u`` and ``v``:

.. code-block:: python

   import pivpy.pivpy  # registers the .piv accessor
   from pivpy import io

   ds = io.create_sample_Dataset(n_frames=1)
   ds = ds.piv.vec2scal('vorticity', name='w')

   grad = ds.piv.gradientf(variable='w')
   # grad contains grad['u'] = dw/dx and grad['v'] = dw/dy

You can plot it like any other vector field:

.. code-block:: python

   grad.piv.showf()

Histogram of a field (PIVMAT-style ``histf``)
=============================================

Compute a histogram for a scalar field (values are stacked over the full field
and all time frames):

.. code-block:: python

   h = ds.piv.histf(variable='w')
   # h is a Dataset with coordinate h['bin'] and counts h['h']

By default, zeros are treated as invalid and excluded. To include zeros, pass
``opt='0'``:

.. code-block:: python

   h0 = ds.piv.histf(variable='w', opt='0')

Vector mode (no variable specified) computes histograms for both components of
the velocity field:

.. code-block:: python

   hv = ds.piv.histf()
   # hv contains hv['hx'] for u (or vx) and hv['hy'] for v (or vy)

Graphical histogram display (PIVMAT-style)
==========================================

PIVPy also includes plotting helpers inspired by PIVMAT's ``histscal_disp`` and
``histvec_disp``.

Scalar histogram display (with optional PDF normalization):

.. code-block:: python

   from pivpy import graphics

   fig, ax = graphics.histscal_disp(ds, variable='w', opt='n')

Vector histogram display:

.. code-block:: python

   fig, ax = graphics.histvec_disp(ds, opt='n')

Extract a rectangular region (PIVMAT-style ``extractf``)
================================================================

Extract a rectangle defined in physical units:

.. code-block:: python

   sub = ds.piv.extractf([0.0, 0.0, 10.0, 5.0], 'phys')  # [x1, y1, x2, y2]

Or extract using mesh indices (MATLAB-like 1-based, inclusive):

.. code-block:: python

   sub = ds.piv.extractf([10, 5, 50, 40], 'mesh')

Optionally retrieve the effective mesh rectangle after clamping:

.. code-block:: python

   sub, mesh_rect = ds.piv.extractf([0.0, 0.0, 10.0, 5.0], 'phys', return_rect=True)

Spatial correlation (PIVMAT-style ``corrm`` and ``corrf``)
================================================================

Matrix correlation along one direction
--------------------------------------

Compute a PIVMAT-like correlation map (returns a ``DataArray`` with a ``lag`` dimension):

.. code-block:: python

   cu = ds.piv.corrm(variable='u', dim='x')
   cv = ds.piv.corrm(variable='v', dim='y', half=True)

Spatial correlation function and integral scales
------------------------------------------------

Compute the 1D correlation function ``f(r)`` and integral scales:

.. code-block:: python

   cor = ds.piv.corrf(variable='u', dim='x', normalize=True)

The result is an ``xarray.Dataset`` with:

- ``cor['r']``: separation length
- ``cor['f']``: correlation function
- scalar diagnostics: ``isinf``, ``r5/is5``, ``r2/is2``, ``r1/is1``, ``r0/is0``

Butterworth filtering (PIVMAT-style ``bwfilterf``)
==================================================

Apply a low-pass Butterworth filter in Fourier space:

.. code-block:: python

   ds_low = ds.piv.bwfilterf(filtsize=3.0, order=8.0, mode='low', trunc=True)

Apply a high-pass filter:

.. code-block:: python

   ds_high = ds.piv.bwfilterf(filtsize=3.0, order=8.0, mode='high')

PIVMAT-compatible option wrapper
--------------------------------

If you prefer PIVMAT-like option strings:

.. code-block:: python

   ds_high2 = ds.piv.bwfilterf_pm(3.0, 8.0, 'high', 'trunc')

Spatial convolution filtering (PIVMAT-style ``filterf``)
========================================================

Apply a Gaussian convolution filter (NaN-aware). Use ``'same'`` to keep the same size:

.. code-block:: python

   ds_smooth = ds.piv.filterf(1.0, 'gauss', 'same')

Or omit ``'same'`` to get Matlab ``conv2(...,'valid')`` behavior (smaller field):

.. code-block:: python

   ds_smooth_valid = ds.piv.filterf(1.0, 'gauss')

Batch processing over filename series (PIVMAT-style ``batchf``)
================================================================

``batchf`` expands bracket patterns (a safe subset of PIVMAT's ``expandstr``)
and processes files one-by-one.

.. code-block:: python

   from pivpy.io import batchf

   # Apply an accessor method by name
   results = batchf(
       'pivpy/data/day2/day2a00500[0:5].T000.D000.P003.H001.L.vec',
       'averf',
   )

You can also pass a callable:

.. code-block:: python

   def mean_speed(ds):
       return ((ds['u'] ** 2 + ds['v'] ** 2) ** 0.5).mean().item()

   speeds = batchf('pivpy/data/day2/day2a00500[0:5].*.vec', mean_speed)

Check for updates on PyPI
=========================

PIVPy can compare the installed version against the latest PyPI version:

.. code-block:: python

   import pivpy

   res = pivpy.check_update(verbose=True)
   # res.status:
   # 0 = server unavailable
   # 1 = up-to-date
   # 2 = update available
   # 3 = installed version newer than PyPI

Notes on running headless
===============================

If you run tests or scripts on a machine without a display, set a non-interactive
Matplotlib backend:

.. code-block:: bash

   MPLBACKEND=Agg uv run pytest -q
