# Convergence analysis

Estimate how many frames are needed for turbulence statistics to converge.

```python
from pivpy import synthetic

ds = synthetic.multivortex(n_frames=16, n=32, n_vortices=6, seed=42)
conv = ds.piv.convergence()
```

```python
from pivpy.graphics import convergence_disp

fig, ax = convergence_disp(conv)
```

![Convergence analysis](convergence_analysis.png)
