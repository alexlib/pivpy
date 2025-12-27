import os

# Ensure a non-interactive backend for headless test runs.
# Must be set before importing matplotlib.pyplot anywhere.
os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_configure(config):
    import matplotlib

    matplotlib.use("Agg", force=True)


def pytest_runtest_teardown(item, nextitem):
    # Always clean up figures to avoid leaking GUI resources.
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass
