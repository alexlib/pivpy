"""Generate static figure assets for the PIVPy Feature Gallery in docs.

Uses a standard reproducible test case: `pivpy.synthetic.multivortex(n_frames=16, n=64, n_vortices=6, seed=42)`
to systematically demonstrate the full breadth of PIVPy features on the same flow physics.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pivpy.pivpy  # registers .piv accessor
from pivpy import synthetic

OUT_DIR = Path("docs_mkdocs/_static/gallery")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Standard Test Case: 16-frame 2D turbulent vortex wake on 64x64 grid
# ---------------------------------------------------------------------------
print("Generating standard test case...")
ds = synthetic.multivortex(n_frames=16, n=64, n_vortices=6, seed=42)
x_mid = float(ds["x"].mean())
y_mid = float(ds["y"].mean())

# 1. Velocity Magnitude & Vectors
print("1. Velocity magnitude...")
fig, ax = ds.piv.plot(background="mag", cmap="viridis", title="Velocity Magnitude |U| & Vector Field", skip=2)
fig.savefig(OUT_DIR / "gallery_vector_magnitude.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 2. Vorticity Field
print("2. Vorticity...")
fig, ax = ds.piv.plot(background="vorticity", cmap="coolwarm", title="Vorticity Field (curl U)", skip=2)
fig.savefig(OUT_DIR / "gallery_vorticity.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 3. Divergence Field
print("3. Divergence...")
fig, ax = ds.piv.plot(background="divergence", cmap="bwr", title="2D Velocity Divergence ∇·u", skip=2)
fig.savefig(OUT_DIR / "gallery_divergence.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 4. Shear Strain Rate
print("4. Shear strain...")
fig, ax = ds.piv.plot(background="strain", cmap="inferno", title="Shear Strain Rate S_xy", skip=2)
fig.savefig(OUT_DIR / "gallery_shear_strain.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 5. Convective Acceleration
print("5. Convective acceleration...")
fig, ax = ds.piv.plot(background="accel", cmap="magma", title="Convective Acceleration |(u·∇)u|", skip=2)
fig.savefig(OUT_DIR / "gallery_acceleration.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 6. Gamma1 Vortex Center Criterion
print("6. Gamma1...")
ds_g1 = ds.piv.gamma1(radius=3, name="gamma1")
fig, ax = ds_g1.piv.plot(background="gamma1", cmap="coolwarm", clim=(-1.0, 1.0), title="Γ₁ Vortex Core Centers", skip=2)
fig.savefig(OUT_DIR / "gallery_gamma1.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 7. Gamma2 Vortex Boundary Criterion
print("7. Gamma2...")
ds_g2 = ds.piv.gamma2(radius=3, name="gamma2")
fig, ax = ds_g2.piv.plot(background="gamma2", cmap="coolwarm", clim=(-1.0, 1.0), title="Γ₂ Galilean-Invariant Vortex Boundaries", skip=2)
fig.savefig(OUT_DIR / "gallery_gamma2.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 8. Q-criterion
print("8. Q-criterion...")
ds_q = ds.piv.q_criterion(name="Q")
fig, ax = ds_q.piv.plot(background="Q", cmap="PiYG", title="Q-Criterion (Rotation vs Strain)", skip=2)
fig.savefig(OUT_DIR / "gallery_q_criterion.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 9. Okubo-Weiss Parameter
print("9. Okubo-Weiss...")
ds_ow = ds.piv.okubo_weiss(name="Q_ow")
fig, ax = ds_ow.piv.plot(background="Q_ow", cmap="PRGn", title="Okubo-Weiss Parameter Q_OW", skip=2)
fig.savefig(OUT_DIR / "gallery_okubo_weiss.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 10. Turbulent Kinetic Energy (TKE)
print("10. TKE...")
ds_tke = ds.piv.tke(name="tke_field", time_average=True)
fig, ax = ds_tke.piv.plot(background="tke_field", cmap="plasma", title="Turbulent Kinetic Energy (TKE)", skip=2)
fig.savefig(OUT_DIR / "gallery_tke.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 11. Reynolds Shear Stress
print("11. Reynolds stress...")
ds_rss = ds.piv.reynolds_stress(name="rey_stress")
fig, ax = ds_rss.piv.plot(background="rey_stress", cmap="bwr", title="Reynolds Shear Stress -<u'v'>", skip=2)
fig.savefig(OUT_DIR / "gallery_reynolds_stress.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 12. Localized Two-Point Spatial Correlation Map
print("12. Two-point correlation...")
corr = ds.piv.two_point_correlation(var_name="u", x_ref=x_mid, y_ref=y_mid, swap_dims=False)
fig, ax = plt.subplots(figsize=(6, 5))
cf = ax.contourf(corr["x"], corr["y"], corr.values, levels=30, cmap="coolwarm")
ax.plot(x_mid, y_mid, "k+", markersize=14, markeredgewidth=2, label=f"Probe ref ({x_mid:.1f}, {y_mid:.1f})")
cb = fig.colorbar(cf, ax=ax, shrink=0.8)
cb.set_label(r"$R_{uu}(x, y; x_{ref}, y_{ref})$", fontsize=11)
ax.set_title(r"Two-Point Probe Correlation $R_{uu}$", fontsize=12, fontweight="bold")
ax.set_xlabel("x [pix]")
ax.set_ylabel("y [pix]")
ax.set_aspect("equal")
ax.legend(loc="upper right", framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT_DIR / "gallery_two_point_corr.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 13. Quadrant Analysis Breakdown
print("13. Quadrant analysis...")
qa = ds.piv.quadrant_analysis(x_var="u", y_var="v")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
quad_labels = [r"$Q_1$ (Outward)", r"$Q_2$ (Ejection)", r"$Q_3$ (Inward)", r"$Q_4$ (Sweep)"]
colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]

# Stress fraction (spatially averaged)
frac_vals = [float(qa["fraction"].sel(quadrant=q).mean().values) for q in range(1, 5)]
ax1.bar(quad_labels, frac_vals, color=colors, alpha=0.85, edgecolor="black")
ax1.set_ylabel("Stress Contribution Fraction")
ax1.set_title(r"Reynolds Stress Contribution $\langle u'v' \rangle_Q / \langle u'v' \rangle$")
ax1.tick_params(axis='x', rotation=20)
ax1.grid(axis='y', linestyle="--", alpha=0.5)

# Residence duration
dur_vals = [float(qa["duration"].sel(quadrant=q).mean().values) for q in range(1, 5)]
ax2.bar(quad_labels, dur_vals, color=colors, alpha=0.85, edgecolor="black")
ax2.set_ylabel("Time / Area Fraction")
ax2.set_title("Event Residence Frequency")
ax2.tick_params(axis='x', rotation=20)
ax2.grid(axis='y', linestyle="--", alpha=0.5)

fig.suptitle("Quadrant Analysis of Turbulent Fluctuations", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "gallery_quadrant_analysis.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 14. Outlier Detection & Cleaning
print("14. Outlier cleaning...")
rng = np.random.default_rng(99)
ds_noisy = ds.copy(deep=True)
mask_out = rng.random(ds_noisy["u"].shape) < 0.05
ds_noisy["u"].values[mask_out] += rng.uniform(-15, 15, size=np.sum(mask_out))
ds_noisy["v"].values[mask_out] += rng.uniform(-15, 15, size=np.sum(mask_out))

ds_cleaned = ds_noisy.piv.clean(threshold=2.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.quiver(ds_noisy.x[::2], ds_noisy.y[::2], ds_noisy["u"].isel(t=0)[::2, ::2], ds_noisy["v"].isel(t=0)[::2, ::2], color="red", scale=80)
ax1.set_title("1. Raw Field with Outliers (5% Spurious)", fontsize=11, fontweight="bold")
ax1.set_aspect("equal")

ax2.quiver(ds_cleaned.x[::2], ds_cleaned.y[::2], ds_cleaned["u"].isel(t=0)[::2, ::2], ds_cleaned["v"].isel(t=0)[::2, ::2], color="navy", scale=80)
ax2.set_title("2. Cleaned (Normalized Median Test + Inpainting)", fontsize=11, fontweight="bold")
ax2.set_aspect("equal")

fig.tight_layout()
fig.savefig(OUT_DIR / "gallery_cleaning_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 15. Turbulent Energy Spectra
print("15. Energy spectra...")
spec = ds.piv.energy_spectrum(radial=True, detrend=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# 2D spectrum
cf = ax1.pcolormesh(spec["kx"], spec["ky"], np.log10(np.maximum(spec["E2D"].values, 1e-10)), cmap="turbo", shading="auto")
fig.colorbar(cf, ax=ax1, label=r"$\log_{10} E(k_x, k_y)$")
ax1.set_title("2D Energy Spectrum", fontsize=11, fontweight="bold")
ax1.set_xlabel(r"$k_x$ [rad/pix]")
ax1.set_ylabel(r"$k_y$ [rad/pix]")
ax1.set_aspect("equal")

# Radial 1D spectrum
k_vals = spec["k"].values[1:]
E_rad = spec["E_radial"].values[1:]
ax2.loglog(k_vals, E_rad, "o-", color="navy", label=r"Measured $E(k)$", linewidth=1.5, markersize=4)
# Reference Kolmogorov slope -5/3
k_ref = np.linspace(k_vals[0], k_vals[-1], 20)
ax2.loglog(k_ref, 0.5 * E_rad[0] * (k_ref / k_vals[0]) ** (-5 / 3), "r--", label=r"$k^{-5/3}$ Kolmogorov line")
ax2.set_title("Radial Turbulent Energy Spectrum", fontsize=11, fontweight="bold")
ax2.set_xlabel(r"Wavenumber $k$ [rad/pix]")
ax2.set_ylabel(r"$E(k)$")
ax2.grid(True, which="both", linestyle="--", alpha=0.5)
ax2.legend()

fig.tight_layout()
fig.savefig(OUT_DIR / "gallery_energy_spectrum.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 16. Pure Streamplot with Magnitude Coloring
print("16. Streamlines...")
speed = np.sqrt(ds["u"].isel(t=0)**2 + ds["v"].isel(t=0)**2).values
fig, ax = ds.piv.streamplot(color=speed, cmap="plasma", density=1.2)
ax.set_title("Flow Streamlines Colored by Velocity Magnitude", fontsize=12, fontweight="bold")
fig.savefig(OUT_DIR / "gallery_streamlines_mag.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("All gallery assets generated successfully in docs_mkdocs/_static/gallery/!")
