import pathlib
import time
from pivpy import io, pivpy, graphics

fieldsDir = pathlib.Path('../../pivpy/data/openpiv_txt/Gamma1_Gamma2_tutorial_notebook')

# Global variables
n = 1

dataBaseXarray = io.load_directory(
    path = fieldsDir,
    basename = "OpenPIVtxtFilePair?",
    ext = ".txt"
)

print("\n Calculating Gamma1")
tic = time.perf_counter()
dataBaseXarray.piv.Γ1(n)
toc = time.perf_counter()
print(f"Time it took to calculate Gamma1: {toc-tic:4f}")

print(dataBaseXarray)