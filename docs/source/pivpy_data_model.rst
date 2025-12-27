=============================
PIVPy data model (xarray)
=============================

PIVPy’s core data structure is an :class:`xarray.Dataset`.
This page explains how it is organized, how to access values correctly,
which metadata is typically stored, and how this compares conceptually
to PIVMat “structures” and “structure arrays”.

.. contents::
   :local:
   :depth: 2


Overview
========

A typical *vector* PIV dataset contains:

- dimensions: ``('y', 'x', 't')``
- coordinates: 1D ``x``, 1D ``y``, 1D ``t``
- data variables: ``u``, ``v`` and ``chc`` (and sometimes ``mask``)

A typical *scalar* dataset contains:

- dimensions: usually ``('y', 'x', 't')`` (or ``('y', 'x')`` for single images)
- data variable: ``w`` (plus optionally ``chc``/``mask``)

The recommended way to interact with PIVPy datasets is via the xarray accessor:

.. code-block:: python

   import pivpy.pivpy  # registers the Dataset.piv accessor
   from pivpy import io

   ds = io.read_piv("B00001.VC7")
   ds.piv.showf()


Single fields
=============

Single vector field
-------------------

A single field is typically represented as a dataset with ``t`` of length 1
(or sometimes without ``t``, depending on the reader and context).

Common data variables
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Vector Dataset variables
   :header-rows: 1
   :widths: 16 24 60

   * - Name
     - Type / dims
     - Meaning
   * - ``u``
     - float, ``(y, x, t)``
     - x-component of velocity (or displacement)
   * - ``v``
     - float, ``(y, x, t)``
     - y-component of velocity (or displacement)
   * - ``chc``
     - int/bool, ``(y, x, t)``
     - validity / choice / quality flag (reader-dependent)
   * - ``mask`` (optional)
     - int/bool, ``(y, x, t)``
     - mask information if present in the source export

Common coordinates
^^^^^^^^^^^^^^^^^^

.. list-table:: Vector Dataset coordinates
   :header-rows: 1
   :widths: 16 24 60

   * - Name
     - Type / dims
     - Meaning
   * - ``x``
     - float, ``(x,)``
     - x-coordinate of grid points (physical units)
   * - ``y``
     - float, ``(y,)``
     - y-coordinate of grid points (physical units)
   * - ``t``
     - float/int, ``(t,)``
     - time coordinate (frames or physical time, depends on loader)

Units and labels
^^^^^^^^^^^^^^^^

PIVPy stores units/labels using xarray attributes:

- ``ds['x'].attrs.get('units')``
- ``ds['y'].attrs.get('units')``
- ``ds['u'].attrs.get('units')``
- ``ds['v'].attrs.get('units')``

Example:

.. code-block:: python

   x_units = ds["x"].attrs.get("units", "")
   u_units = ds["u"].attrs.get("units", "")


Single scalar field
-------------------

Scalar fields (images or derived scalars like vorticity) usually use ``w``:

.. list-table:: Scalar Dataset variables
   :header-rows: 1
   :widths: 16 24 60

   * - Name
     - Type / dims
     - Meaning
   * - ``w``
     - float, ``(y, x, t)`` or ``(y, x)``
     - scalar field values (e.g. vorticity, divergence, intensity)

Scalar datasets are often produced using ``vec2scal``:

.. code-block:: python

   import pivpy.pivpy

   vort = ds.piv.vec2scal("vorticity", name="w")
   vort.piv.showf(scalar="w")


Time series (“array of fields”)
===============================

In PIVMat, a “structure array” stores many fields as a 1D array of structs.
In PIVPy, the usual equivalent is a *single* :class:`xarray.Dataset` where
multiple frames are stacked along the ``t`` dimension.

Inspecting shape
----------------

.. code-block:: python

   ds.dims
   ds.sizes  # contains sizes for y/x/t

Selecting frames
----------------

.. code-block:: python

   first = ds.isel(t=0)
   one_frame = ds.isel(t=5)
   subset = ds.isel(t=slice(0, 10))


Indexing and coordinate conventions
===================================

xarray uses NumPy-like indexing: arrays stored as ``(y, x, t)`` mean:

- axis 0 is y (rows)
- axis 1 is x (columns)
- axis 2 is time

So the element access is:

.. code-block:: python

   # value at y-index iy, x-index ix, time-index it
   val = ds["u"].values[iy, ix, it]

Coordinate-based selection is also available:

.. code-block:: python

   # select nearest physical coordinate
   pt = ds.sel(x=10.0, y=20.0, method="nearest")

Plotting conventions
--------------------

PIVPy plotting helpers are designed to respect coordinate arrays.
For example:

.. code-block:: python

   import pivpy.pivpy

   ds.isel(t=0).piv.showf(background="vorticity")


About “ysign”
-------------

PIVMat stores a dedicated ``ysign`` string to track upward vs downward y-axis.
PIVPy generally relies on the *ordering of the coordinate arrays* ``y`` and ``x``.
The plotting functions in :mod:`pivpy.graphics` handle flipped axes by inspecting
the coordinate ordering.


Metadata (attrs)
================

PIVPy stores dataset-level metadata in ``ds.attrs``.
Not every reader populates the same keys, but you will often see entries like:

- ``name``: file name
- ``setname``: directory / set identifier
- ``source``: origin/format
- ``delta_t``: time step (when known)
- ``history``: list of processing steps (PIVMat-like concept)

Example:

.. code-block:: python

   ds.attrs
   ds.attrs.get("history", [])

Many PIVPy operations append to ``ds.attrs['history']`` (when the function is
written in the PIVMat-compat style).


DaVis / vendor attributes
=========================

PIVMat stores vendor metadata in a big string field called ``Attributes`` and uses
functions like ``getattribute`` to parse it.

In PIVPy, vendor-specific metadata may appear either:

- in ``ds.attrs`` as structured keys, or
- as separate variables/attributes (reader-dependent)

The convenience functions in :mod:`pivpy.io` and the accessor methods in
:class:`pivpy.pivpy.PIVAccessor` are the recommended way to work with those details.


Practical examples
==================

Show the dataset “layout”
-------------------------

.. code-block:: python

   import pivpy.pivpy
   from pivpy import io

   ds = io.read_piv("B00001.VC7")

   print(ds)
   print("dims:", ds.dims)
   print("vars:", list(ds.data_vars))
   print("coords:", list(ds.coords))

Compute a derived scalar with consistent units
----------------------------------------------

.. code-block:: python

   vort = ds.piv.vec2scal("vorticity", name="w")
   print(vort["w"].attrs.get("units", ""))

Track processing history
------------------------

.. code-block:: python

   ds2 = ds.piv.filterf(1.0, "gauss")
   ds2 = ds2.piv.vec2scal("vorticity", name="w")
   print(ds2.attrs.get("history", []))

