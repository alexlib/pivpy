import importlib.resources
import importlib.util

import pytest

from pivpy import io, schema

try:
    path = importlib.resources.files('pivpy') / 'data'
except NameError:  # pragma: no cover
    with importlib.resources.path('pivpy', 'data') as data_path:
        path = data_path

VEC_FILE = path / "Insight" / "Run000002.T000.D000.P000.H001.L.vec"
OPENPIV_FILE = path / "openpiv_txt" / "exp1_001_b.txt"
DAVIS8_FILE = path / "PIV_Challenge" / "B00001.txt"
PIVLAB_FILE = path / "pivlab" / "test_pivlab.mat"
VC7_FILE = path / "urban_canopy" / "B00001.vc7"


@pytest.mark.parametrize(
    "reader_cls,filepath",
    [
        (io.InsightVECReader, VEC_FILE),
        (io.OpenPIVReader, OPENPIV_FILE),
        (io.Davis8Reader, DAVIS8_FILE),
        (io.PIVLabReader, PIVLAB_FILE),
    ],
)
def test_reader_output_matches_schema(reader_cls, filepath):
    if reader_cls is io.PIVLabReader and importlib.util.find_spec("h5py") is None:
        pytest.skip("h5py not installed")
    ds = reader_cls().read(filepath)
    schema.validate(ds)
    assert schema.is_valid(ds)
    assert ds.attrs["pivpy_schema_version"] == schema.SCHEMA_VERSION


def test_vc7_reader_output_matches_schema():
    pytest.importorskip("lvpyio")
    ds = io.LaVisionVC7Reader().read(VC7_FILE)
    schema.validate(ds)


def test_openpiv_reader_folds_mask_into_chc():
    mask_file = path / "openpiv_txt" / "Gamma1_Gamma2_tutorial_notebook" / "OpenPIVtxtFilePair0.txt"
    ds = io.OpenPIVReader().read(mask_file)
    assert "mask" not in ds
    schema.validate(ds)


def test_validate_rejects_missing_variable():
    ds = io.create_sample_Dataset()
    with pytest.raises(ValueError, match="missing required variables"):
        schema.validate(ds.drop_vars("chc"))


def test_is_valid_false_for_bad_dataset():
    ds = io.create_sample_Dataset()
    assert schema.is_valid(ds.drop_vars("chc")) is False
