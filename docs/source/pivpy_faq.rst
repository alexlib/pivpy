====================
PIVPy FAQ
====================

This page is a practical, example-driven FAQ for common PIVPy workflows.
It is inspired by the kind of questions users ask when coming from MATLAB toolboxes,
but the snippets below use real PIVPy functions and the xarray-based PIVPy data model.

Most examples assume you have installed optional dependencies if you need them
(e.g. `lvpyio` for LaVision `.vc7`).

.. contents::
   :local:
   :depth: 1


Basic questions
===============

.. _faq-import-display:

How do I import and display velocity fields?
--------------------------------------------

Load a single file (format auto-detected) and display it:

.. code-block:: python

   import pivpy.pivpy  # registers the Dataset.piv accessor
   from pivpy import io

   ds = io.read_piv("B00001.VC7")
   ds.piv.showf()  # vector field

Load many files (directory / glob) and display the first frame:

.. code-block:: python

   ds = io.loadvec("*.VC7")  # or io.read_directory("path/to/dir")
   ds.isel(t=0).piv.showf()


.. _faq-help:

How do I get quick help on a PIVPy function?
---------------------------------------------

In Python you can use `help()` (or IPython `?`):

.. code-block:: python

   import pivpy
   from pivpy import io

   help(io.read_piv)
   help(io.loadvec)

For API-level docs, see :doc:`api_reference`.


.. _faq-smooth:

My velocity fields are noisy. How do I smooth them?
---------------------------------------------------

Two common choices:

1) A spatial convolution filter (Gaussian / kernel-based), similar in spirit to PIVMat’s filtering:

.. code-block:: python

   import pivpy.pivpy
   from pivpy import io

   ds = io.read_piv("B00001.VC7")
   ds_smooth = ds.piv.filterf(1.0, "gauss")
   ds_smooth.piv.showf()

2) A frequency-domain Butterworth filter:

.. code-block:: python

   ds_low = ds.piv.bwfilterf(filtsize=3.0, order=8.0, mode="low")
   ds_low.piv.showf()

If you specifically want a *temporal running average* over frames, use `smoothf`:

.. code-block:: python

   from pivpy.compute_funcs import smoothf

   ds_smoothed_in_time = smoothf(ds, n=5)


.. _faq-vorticity:

How do I display vorticity from a velocity field?
-------------------------------------------------

Option A: compute a scalar field and display it:

.. code-block:: python

   import pivpy.pivpy
   from pivpy import io

   ds = io.read_piv("B00001.VC7")
   vort = ds.piv.vec2scal("vorticity", name="w")
   vort.piv.showf(scalar="w")

Option B: use a scalar background behind the vectors:

.. code-block:: python

   ds.piv.showf(background="vorticity")


.. _faq-first-n:

I only want to view the first N frames
--------------------------------------

With a time-series dataset, select a slice on `t`:

.. code-block:: python

   ds10 = ds.isel(t=slice(0, 10))
   ds10.isel(t=0).piv.showf()

To create a quick movie (mp4/gif depending on your Matplotlib/ffmpeg setup):

.. code-block:: python

   ds10.piv.to_movie("preview.mp4", background="vorticity")


.. _faq-load-images:

Can I import IMX/IM7 image files?
---------------------------------

PIVPy includes readers for several formats; availability depends on optional dependencies.
If you have `.imx`/`.im7` support in your environment, you can load them as scalar datasets
and display them:

.. code-block:: python

   import pivpy.pivpy
   from pivpy import io

   img = io.openim7("frame.im7")
   img.piv.showf(scalar="w", cmap="gray")


.. _faq-load-txt:

Can I import vector fields saved in TXT/DAT?
--------------------------------------------

Yes. If you have OpenPIV/ASCII-like exports:

.. code-block:: python

   import pivpy.pivpy
   from pivpy import io

   ds = io.load_openpiv_txt("field.txt")
   ds.piv.showf()


.. _faq-vectors-missing:

Why aren’t all vectors displayed?
---------------------------------

By default, dense fields can look unreadable. You can subsample arrows in the quiver plot
using `nthArr` (plot every Nth vector in each direction):

.. code-block:: python

   ds.piv.showf(nthArr=3)


.. _faq-readimx:

What’s the relationship between LaVision’s ReadIMX and PIVPy?
--------------------------------------------------------------

PIVPy can use `lvpyio` (when installed) to read LaVision VC7/IMX/IM7 content.
PIVPy then stores the result in a consistent `xarray.Dataset` and provides
processing/plotting via the `Dataset.piv` accessor.


Advanced questions
==================

.. _faq-isolines:

How do I plot iso-lines (contours) of a scalar property?
--------------------------------------------------------

Compute a scalar (e.g., velocity magnitude) and use Matplotlib contours:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   import pivpy.pivpy

   mag = ds.piv.vec2scal("norm", name="w")
   w = np.asarray(mag["w"].isel(t=0).values)

   X, Y = np.meshgrid(mag["x"].values, mag["y"].values)
   fig, ax = plt.subplots()
   ax.contourf(X, Y, w, levels=15)
   ax.set_aspect("equal")


.. _faq-navigate:

How do I “navigate” through frames?
-----------------------------------

Because fields are stored as an xarray time series, you can index frames with `isel`:

.. code-block:: python

   frame = ds.isel(t=5)
   frame.piv.showf(background="vorticity")

Or iterate:

.. code-block:: python

   for ti in range(ds.sizes["t"]):
       ds.isel(t=ti).piv.showf()


.. _faq-data-model:

How are fields stored in PIVPy?
-------------------------------

PIVPy’s core structure is an `xarray.Dataset` with:

- dimensions: `('y', 'x', 't')`
- coordinates: `x`, `y`, `t`
- variables: usually `u`, `v`, and `chc` (validity)

Example:

.. code-block:: python

   list(ds.dims)
   list(ds.data_vars)
   ds


.. _faq-profiles:

How do I plot a velocity profile or a vorticity profile?
--------------------------------------------------------

Pick a line index and plot with Matplotlib:

.. code-block:: python

   import matplotlib.pyplot as plt

   j = 12  # y-index
   uline = ds["u"].isel(t=0, y=j)
   plt.plot(ds["x"], uline)
   plt.xlabel("x")
   plt.ylabel("u")

For a vorticity profile:

.. code-block:: python

   vort = ds.piv.vec2scal("vorticity", name="w")
   wline = vort["w"].isel(t=0, y=j)
   plt.plot(vort["x"], wline)


.. _faq-spatiotemp-diagram:

How do I plot a spatio-temporal diagram?
----------------------------------------

Build a matrix where x is one axis and time is the other:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   j = 12
   d = np.asarray(ds["u"].isel(y=j).values).T  # shape (t, x)

   plt.figure()
   plt.imshow(d, aspect="auto", origin="lower")
   plt.xlabel("x-index")
   plt.ylabel("t-index")

Or, for a PIVMat-style spatio-temporal correlation on a scalar series, use `spatiotempcorrf`:

.. code-block:: python

   from pivpy.compute_funcs import spatiotempcorrf

   scalar = ds.piv.vec2scal("vorticity", name="w")
   cor = spatiotempcorrf(scalar)


.. _faq-history:

How do I see the list of operations applied to a field?
--------------------------------------------------------

Many PIVPy operations append entries to `ds.attrs['history']`.

.. code-block:: python

   ds2 = ds.piv.filterf(1.0, "gauss")
   ds2.attrs.get("history", [])


Expert questions
================

.. _faq-memory:

I have hundreds/thousands of fields and can’t keep them all in memory
---------------------------------------------------------------------

If your workflow can be expressed as “load a file → process → reduce/save”, use `batchf`:

.. code-block:: python

   from pivpy.io import batchf

   # Apply an accessor method to each file and collect results
   results = batchf("*.VC7", "averf")

For custom logic, iterate over filenames and process one at a time.


.. _faq-export:

How do I save processed fields?
--------------------------------

Common choices:

- NetCDF via xarray:

.. code-block:: python

   ds.to_netcdf("processed.nc")

- MATLAB `.mat` export helper for interoperability:

.. code-block:: python

   from pivpy.io import vec2mat

   vec2mat(ds, "processed.mat")


.. _faq-rms:

How do I compute RMS over a time series?
----------------------------------------

`averf` can return average, std, and rms:

.. code-block:: python

   avg, std, rms = ds.piv.averf(return_std_rms=True)
   rms.piv.showf(background=None)

