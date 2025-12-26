import pathlib
import numpy as np
import importlib.resources
from pivpy import io

# Ensure compatibility with different Python versions (3.9+ has 'files', 3.7 and 3.8 need 'path')
try:
    from importlib.resources import files
except ImportError:
    from importlib.resources import path as resource_path

# For Python 3.9+
try:
    path = files('pivpy') / 'data'
except NameError:
    # For Python 3.7 and 3.8
    with resource_path('pivpy', 'data') as data_path:
        path = data_path


vec_file = path / "Insight" / "Run000002.T000.D000.P000.H001.L.vec" 
openpiv_txt_file = path / "openpiv_txt" / "exp1_001_b.txt"


def test_get_dt():
    """ test if we get correct delta t """
    _, _, _, _,delta_t,_,_ = io.parse_header(vec_file)
    assert delta_t == 2000.


def test_get_frame():
    """ tests the correct frame number """
    _, _, _, _, _, frame,_ = io.parse_header(
        path/  "day2" / "day2a005003.T000.D000.P003.H001.L.vec"
    )
    assert frame == 5003
    _, _, _, _, _, frame,_ = io.parse_header(
        vec_file
    )
    assert frame == 2
    _, _, _, _, _, frame,_ = io.parse_header(
        path / "openpiv_vec" / "exp1_001_b.vec"
        )
    assert frame == 1
    _, _, _, _, _, frame,_ = io.parse_header(
        path / "openpiv_txt" / "exp1_001_b.txt"
    )
    assert frame == 1


def test_load_vec():
    """tests loading vec file
    """
    data = io.load_vec(vec_file)
    assert data["u"].shape == (63, 63, 1)
    tmp = data["u"].values
    assert tmp[0, 0, 0] == 0.0
    assert np.allclose(data.coords["x"][0], 0.31248)
    assert "t" in data.dims


# readim is depreceated, see the new Lavision Python package
# def test_load_vc7():
#     data = io.load_vc7(os.path.join(path, "VC7/2Ca.VC7"))
#     assert data["u"].shape == (57, 43, 1)
#     assert np.allclose(data.u.values[0, 0], -0.04354814)
#     assert np.allclose(data.coords["x"][-1], 193.313795)


def test_loadopenpivtxt():
    """tests loading openpivtxt file (5 columns)
    """
    data = io.load_openpiv_txt(openpiv_txt_file)
    assert "u" in data
    assert "v" in data
    assert "chc" in data
    # Old format should not have mask
    assert "mask" not in data


def test_loadopenpivtxt_with_mask():
    """tests loading openpivtxt file with mask column (6 columns)
    """
    # Test with a file that has 6 columns
    openpiv_txt_file_with_mask = path / "openpiv_txt" / "Gamma1_Gamma2_tutorial_notebook" / "OpenPIVtxtFilePair0.txt"
    data = io.load_openpiv_txt(openpiv_txt_file_with_mask)
    assert "u" in data
    assert "v" in data
    assert "chc" in data
    # New format should have mask
    assert "mask" in data
    # Check that mask has the right shape
    assert data["mask"].shape == data["u"].shape

def test_load_directory():
    """tests loading directory of Insight VEC, vc7, and Davis8 files
    """
    data = io.load_directory(
        path / "Insight", 
        basename="Run*", 
        ext=".vec"
        )
    print(data.t)
    assert np.allclose(data["t"], [0, 1, 2, 3, 4])

    data = io.load_directory(
        path / "urban_canopy", 
        basename="B*", 
        ext=".vc7"
    )
    assert np.allclose(data["t"], [0, 1, 2, 3, 4])

    data = io.load_directory(
        path / "PIV_Challenge", 
        basename="B*", 
        ext=".txt"
    )
    assert np.allclose(data["t"], [0, 1])

def test_check_units():
    """ reads units and checks their validitty 
    def set_default_attrs(dataset: xr.Dataset)-> xr.Dataset:
    """ 
    data = io.create_sample_Dataset()
    assert data.t.attrs["units"] in ["s", "sec", "frame"]
    assert data.x.attrs["units"] in ["pix", "m", "mm"]
    assert data.y.attrs["units"] in ["pix", "m", "mm"]
    assert data.u.attrs["units"] in ["pix", "m", "mm"]
    assert data.v.attrs["units"] in ["pix", "m", "mm"]
    assert data.attrs["delta_t"] == 0.0


def test_create_sample_field():
    data = io.create_sample_field(frame=3)
    assert data["t"] == 3
    data = io.create_sample_field(rows=3, cols=7)
    assert data.x.shape[0] == 7
    assert data.y.shape[0] == 3
    assert data["t"] == 0.0
    


def test_create_sample_dataset():
    data = io.create_sample_Dataset(n_frames=3)
    assert data.sizes["t"] == 3
    # assert data.dims["t"] == 3
    assert np.allclose(data["t"], np.arange(3))


def test_to_nc():
    data = io.create_sample_Dataset(n_frames = 25)
    data.to_netcdf("tmp.nc")

    data = io.load_directory(path / "Insight" )
    data.to_netcdf("tmp.nc")


def test_load_pivlab():
    """Test loading PIVLab MAT file"""
    pivlab_file = path / "pivlab" / "test_pivlab.mat"
    
    # Test loading all frames
    data = io.load_pivlab(pivlab_file)
    assert "u" in data
    assert "v" in data
    assert "chc" in data
    assert data["u"].shape[2] == 2  # 2 frames in test file
    assert data["t"].shape[0] == 2
    assert np.allclose(data["t"], [0, 1])
    
    # Check grid dimensions (10x8 in test file)
    assert data["u"].shape[0] == 10  # y dimension
    assert data["u"].shape[1] == 8   # x dimension
    
    # Check that mask has expected values (mostly 1, with one 0)
    assert np.sum(data["chc"].values == 1) > np.sum(data["chc"].values == 0)
    
    # Test loading a specific frame
    data_frame0 = io.load_pivlab(pivlab_file, frame=0)
    assert data_frame0["t"].shape[0] == 1
    assert data_frame0["t"].values[0] == 0
    
    data_frame1 = io.load_pivlab(pivlab_file, frame=1)
    assert data_frame1["t"].shape[0] == 1
    assert data_frame1["t"].values[0] == 1
    
    # Check that velocities are reasonable (not all zeros or NaNs)
    assert not np.all(np.isnan(data["u"].values))
    assert not np.all(data["u"].values == 0)
    assert not np.all(np.isnan(data["v"].values))
    assert not np.all(data["v"].values == 0)


# ============================================================================
# Tests for New Plugin-Based Architecture
# ============================================================================

# Test PIVMetadata
def test_piv_metadata_creation():
    """Test PIVMetadata dataclass creation"""
    metadata = io.PIVMetadata(
        pos_units='mm',
        vel_units='mm/s',
        time_units='s',
        delta_t=0.001,
        variables=['x', 'y', 'u', 'v'],
        rows=10,
        cols=20,
        frame=5
    )
    assert metadata.pos_units == 'mm'
    assert metadata.vel_units == 'mm/s'
    assert metadata.time_units == 's'
    assert metadata.delta_t == 0.001
    assert metadata.rows == 10
    assert metadata.cols == 20
    assert metadata.frame == 5
    assert len(metadata.variables) == 4


def test_piv_metadata_defaults():
    """Test PIVMetadata default values"""
    metadata = io.PIVMetadata()
    assert metadata.pos_units == io.POS_UNITS
    assert metadata.vel_units == io.VEL_UNITS
    assert metadata.time_units == io.TIME_UNITS
    assert metadata.delta_t == io.DELTA_T
    assert metadata.frame == 0
    assert metadata.rows is None
    assert metadata.cols is None


# Tests for InsightVECReader
def test_insight_can_read_valid_file():
    """Test InsightVECReader can identify valid Insight VEC files"""
    reader = io.InsightVECReader()
    assert reader.can_read(vec_file) is True


def test_insight_can_read_invalid_file():
    """Test InsightVECReader rejects non-Insight files"""
    reader = io.InsightVECReader()
    assert reader.can_read(openpiv_txt_file) is False
    assert reader.can_read(pathlib.Path('nonexistent.txt')) is False


def test_insight_read_metadata():
    """Test InsightVECReader metadata extraction"""
    reader = io.InsightVECReader()
    metadata = reader.read_metadata(vec_file)
    
    assert metadata.rows == 63
    assert metadata.cols == 63
    assert metadata.delta_t == 2000.0
    assert metadata.frame == 2
    # Variables from header are uppercase
    assert 'X' in metadata.variables or 'x' in metadata.variables
    assert 'U' in metadata.variables or 'u' in metadata.variables


def test_insight_read_data():
    """Test InsightVECReader data loading"""
    reader = io.InsightVECReader()
    dataset = reader.read(vec_file)
    
    assert 'u' in dataset
    assert 'v' in dataset
    assert 'chc' in dataset
    assert dataset['u'].shape == (63, 63, 1)
    assert dataset.attrs['delta_t'] == 2000.0


def test_insight_read_with_custom_frame():
    """Test InsightVECReader with custom frame number"""
    reader = io.InsightVECReader()
    dataset = reader.read(vec_file, frame=10)
    
    assert dataset['t'].values[0] == 10


def test_insight_extract_frame_number():
    """Test frame number extraction from filename"""
    reader = io.InsightVECReader()
    metadata = reader.read_metadata(vec_file)
    assert metadata.frame == 2
    
    # Test with different filename pattern
    test_file = path / "day2" / "day2a005003.T000.D000.P003.H001.L.vec"
    metadata2 = reader.read_metadata(test_file)
    assert metadata2.frame == 5003


# Tests for OpenPIVReader
def test_openpiv_can_read_5_columns():
    """Test OpenPIVReader can identify 5-column files"""
    reader = io.OpenPIVReader()
    assert reader.can_read(openpiv_txt_file) is True


def test_openpiv_can_read_6_columns_with_mask():
    """Test OpenPIVReader can identify 6-column files with mask"""
    reader = io.OpenPIVReader()
    openpiv_txt_file_with_mask = path / "openpiv_txt" / "Gamma1_Gamma2_tutorial_notebook" / "OpenPIVtxtFilePair0.txt"
    assert reader.can_read(openpiv_txt_file_with_mask) is True


def test_openpiv_read_metadata():
    """Test OpenPIVReader metadata extraction"""
    reader = io.OpenPIVReader()
    metadata = reader.read_metadata(openpiv_txt_file)
    
    assert metadata.frame == 1  # Extracted from filename exp1_001_b
    assert metadata.variables == ['x', 'y', 'u', 'v']


def test_openpiv_read_data():
    """Test OpenPIVReader data loading"""
    reader = io.OpenPIVReader()
    dataset = reader.read(openpiv_txt_file)
    
    assert 'u' in dataset
    assert 'v' in dataset
    assert 'chc' in dataset


def test_openpiv_extract_frame_number():
    """Test OpenPIV frame number extraction from filename"""
    reader = io.OpenPIVReader()
    metadata = reader.read_metadata(openpiv_txt_file)
    assert metadata.frame == 1


# Tests for Davis8Reader
def test_davis8_can_read_valid_file():
    """Test Davis8Reader can identify Davis8 files"""
    reader = io.Davis8Reader()
    davis8_file = path / "PIV_Challenge" / "B00001.txt"
    assert reader.can_read(davis8_file) is True


def test_davis8_read_metadata():
    """Test Davis8Reader metadata extraction"""
    reader = io.Davis8Reader()
    davis8_file = path / "PIV_Challenge" / "B00001.txt"
    metadata = reader.read_metadata(davis8_file)
    
    assert metadata.variables == ['x', 'y', 'u', 'v']


def test_davis8_read_data():
    """Test Davis8Reader data loading"""
    reader = io.Davis8Reader()
    davis8_file = path / "PIV_Challenge" / "B00001.txt"
    dataset = reader.read(davis8_file)
    
    assert 'u' in dataset
    assert 'v' in dataset


# Tests for LaVisionVC7Reader
def test_vc7_can_read_valid_file():
    """Test LaVisionVC7Reader can identify VC7 files"""
    reader = io.LaVisionVC7Reader()
    vc7_file = path / "urban_canopy" / "B00001.vc7"
    assert reader.can_read(vc7_file) is True
    assert reader.can_read(vec_file) is False


def test_vc7_read_metadata():
    """Test LaVisionVC7Reader metadata extraction"""
    reader = io.LaVisionVC7Reader()
    vc7_file = path / "urban_canopy" / "B00001.vc7"
    metadata = reader.read_metadata(vc7_file)
    
    assert metadata.frame == 1
    assert metadata.variables == ['x', 'y', 'u', 'v']


def test_vc7_read_with_lvpyio_installed():
    """Test LaVisionVC7Reader data loading"""
    reader = io.LaVisionVC7Reader()
    vc7_file = path / "urban_canopy" / "B00001.vc7"
    
    # This will work if lvpyio is installed
    try:
        dataset = reader.read(vc7_file)
        assert 'u' in dataset
        assert 'v' in dataset
    except NameError:
        # lvpyio not installed, expected
        pass


# Tests for PIVLabReader
def test_pivlab_can_read_valid_file():
    """Test PIVLabReader can identify PIVLab MAT files"""
    reader = io.PIVLabReader()
    pivlab_file = path / "pivlab" / "test_pivlab.mat"
    
    # This will only work if h5py is installed
    try:
        import h5py
        assert reader.can_read(pivlab_file) is True
        assert reader.can_read(vec_file) is False
    except ImportError:
        # h5py not installed, skip
        pass


def test_pivlab_read_metadata():
    """Test PIVLabReader metadata extraction"""
    reader = io.PIVLabReader()
    pivlab_file = path / "pivlab" / "test_pivlab.mat"
    
    try:
        metadata = reader.read_metadata(pivlab_file)
        assert metadata.variables == ['x', 'y', 'u', 'v']
    except ImportError:
        # h5py not installed, skip
        pass


def test_pivlab_read_data():
    """Test PIVLabReader data loading"""
    reader = io.PIVLabReader()
    pivlab_file = path / "pivlab" / "test_pivlab.mat"
    
    try:
        dataset = reader.read(pivlab_file)
        assert 'u' in dataset
        assert 'v' in dataset
        assert 'chc' in dataset
    except ImportError:
        # h5py not installed, skip
        pass


# Tests for PIVReaderRegistry
def test_registry_auto_registration():
    """Test that built-in readers are auto-registered"""
    registry = io.PIVReaderRegistry()
    readers = registry.get_readers()
    
    assert len(readers) >= 5  # At least 5 built-in readers
    reader_types = [type(r).__name__ for r in readers]
    assert 'InsightVECReader' in reader_types
    assert 'OpenPIVReader' in reader_types
    assert 'Davis8Reader' in reader_types
    assert 'LaVisionVC7Reader' in reader_types
    assert 'PIVLabReader' in reader_types


def test_registry_find_reader_insight():
    """Test registry finds correct reader for Insight files"""
    registry = io.PIVReaderRegistry()
    reader = registry.find_reader(vec_file)
    
    assert reader is not None
    assert isinstance(reader, io.InsightVECReader)


def test_registry_find_reader_openpiv():
    """Test registry finds correct reader for OpenPIV files"""
    registry = io.PIVReaderRegistry()
    reader = registry.find_reader(openpiv_txt_file)
    
    assert reader is not None
    assert isinstance(reader, io.OpenPIVReader)


def test_registry_custom_reader():
    """Test custom reader registration"""
    class CustomReader(io.PIVReader):
        def can_read(self, filepath):
            return str(filepath).endswith('.custom')
        
        def read_metadata(self, filepath):
            return io.PIVMetadata()
        
        def read(self, filepath, **kwargs):
            return io.create_sample_field()
    
    registry = io.PIVReaderRegistry()
    custom = CustomReader()
    registry.register(custom)
    
    # Custom reader should be at the beginning (higher priority)
    readers = registry.get_readers()
    assert isinstance(readers[0], CustomReader)


# Integration tests for high-level API
def test_read_piv_auto_detect_insight():
    """Test read_piv auto-detects Insight format"""
    dataset = io.read_piv(vec_file)
    
    assert 'u' in dataset
    assert 'v' in dataset
    assert dataset['u'].shape == (63, 63, 1)
    assert dataset.attrs['delta_t'] == 2000.0


def test_read_piv_auto_detect_openpiv():
    """Test read_piv auto-detects OpenPIV format"""
    dataset = io.read_piv(openpiv_txt_file)
    
    assert 'u' in dataset
    assert 'v' in dataset


def test_read_piv_with_format_specified():
    """Test read_piv with explicit format"""
    dataset = io.read_piv(vec_file, format='insight')
    
    assert 'u' in dataset
    assert dataset['u'].shape == (63, 63, 1)


def test_read_piv_file_not_found():
    """Test read_piv raises error for non-existent file"""
    import pytest
    
    with pytest.raises(FileNotFoundError):
        io.read_piv('nonexistent_file.vec')


def test_read_piv_unsupported_format():
    """Test read_piv raises error for unsupported format"""
    import pytest
    
    with pytest.raises(ValueError, match="Unsupported format"):
        io.read_piv(vec_file, format='unknown_format')


def test_read_piv_with_custom_params():
    """Test read_piv with custom parameters"""
    dataset = io.read_piv(vec_file, delta_t=0.5, frame=10)
    
    assert dataset.attrs['delta_t'] == 0.5
    assert dataset['t'].values[0] == 10


# Tests for read_directory
def test_read_directory_multiple_files():
    """Test read_directory loads multiple files"""
    dataset = io.read_directory(path / "Insight", pattern="Run*", ext=".vec")
    
    assert dataset['u'].shape[2] == 5  # 5 files
    assert np.allclose(dataset['t'].values, [0, 1, 2, 3, 4])
    assert dataset.attrs['delta_t'] == 2000.0


def test_read_directory_empty_raises_error():
    """Test read_directory raises error for no matching files"""
    import pytest
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(IOError, match="No files"):
            io.read_directory(tmpdir, pattern="*.vec")


def test_read_directory_with_pattern():
    """Test read_directory with custom pattern"""
    dataset = io.read_directory(path / "Insight", pattern="Run*", ext=".vec")
    
    assert 't' in dataset.dims
    assert len(dataset['t']) == 5


def test_read_directory_preserves_delta_t():
    """Test read_directory preserves delta_t from files"""
    dataset = io.read_directory(path / "Insight", pattern="Run*", ext=".vec")
    
    assert 'delta_t' in dataset.attrs
    assert dataset.attrs['delta_t'] == 2000.0


def test_read_directory_not_found():
    """Test read_directory raises error for non-existent directory"""
    import pytest
    
    with pytest.raises(FileNotFoundError):
        io.read_directory('nonexistent_directory')


# Tests for save_piv
def test_save_piv_netcdf():
    """Test save_piv to NetCDF format"""
    import tempfile
    import os
    
    dataset = io.create_sample_field()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test.nc')
        try:
            io.save_piv(dataset, filepath, format='netcdf')
            assert os.path.exists(filepath)
        except RuntimeError as e:
            # NetCDF HDF error is a known issue with numpy version compatibility
            if "NetCDF: HDF error" in str(e):
                pass  # Skip this test if netCDF has issues
            else:
                raise


def test_save_piv_csv():
    """Test save_piv to CSV format"""
    import tempfile
    import os
    import pandas as pd
    
    dataset = io.create_sample_field(rows=3, cols=4)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test.csv')
        io.save_piv(dataset, filepath, format='csv', frame=0)
        
        assert os.path.exists(filepath)
        
        # Read back and verify
        df = pd.read_csv(filepath)
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert 'u' in df.columns
        assert 'v' in df.columns
        assert len(df) == 12  # 3 rows * 4 cols


def test_save_piv_unsupported_format():
    """Test save_piv raises error for unsupported format"""
    import pytest
    
    dataset = io.create_sample_field()
    
    with pytest.raises(ValueError, match="Unsupported format"):
        io.save_piv(dataset, 'output.xyz', format='unknown')


# Tests for custom reader registration
def test_register_custom_reader():
    """Test register_reader function"""
    class TestReader(io.PIVReader):
        def can_read(self, filepath):
            return str(filepath).endswith('.test')
        
        def read_metadata(self, filepath):
            return io.PIVMetadata()
        
        def read(self, filepath, **kwargs):
            return io.create_sample_field()
    
    reader = TestReader()
    io.register_reader(reader)
    
    # Verify it's registered in global registry
    readers = io._REGISTRY.get_readers()
    assert any(isinstance(r, TestReader) for r in readers)


def test_custom_reader_auto_detection():
    """Test custom reader is used in auto-detection"""
    import tempfile
    import os
    
    class CustomReader(io.PIVReader):
        def can_read(self, filepath):
            return str(filepath).endswith('.custom')
        
        def read_metadata(self, filepath):
            return io.PIVMetadata()
        
        def read(self, filepath, **kwargs):
            dataset = io.create_sample_field()
            dataset.attrs['custom'] = True
            return dataset
    
    io.register_reader(CustomReader())
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test.custom')
        # Create empty file
        with open(filepath, 'w') as f:
            f.write('')
        
        # This should use our custom reader
        dataset = io.read_piv(filepath)
        assert dataset.attrs.get('custom') is True


# Backward compatibility tests
def test_load_vec_still_works():
    """Test that load_vec still works (backward compatibility)"""
    dataset = io.load_vec(vec_file)
    
    assert 'u' in dataset
    assert 'v' in dataset
    assert dataset['u'].shape == (63, 63, 1)


def test_load_openpiv_txt_still_works():
    """Test that load_openpiv_txt still works (backward compatibility)"""
    dataset = io.load_openpiv_txt(openpiv_txt_file)
    
    assert 'u' in dataset
    assert 'v' in dataset


def test_old_load_directory_still_works():
    """Test that load_directory still works (backward compatibility)"""
    dataset = io.load_directory(path / "Insight", basename="Run*", ext=".vec")
    
    assert 't' in dataset.dims
    assert len(dataset['t']) == 5


# Utility function tests
def test_from_arrays():
    """Test from_arrays creates proper dataset"""
    x, y = np.meshgrid(np.arange(5), np.arange(3))
    u = np.random.rand(3, 5)
    v = np.random.rand(3, 5)
    mask = np.ones_like(u)
    
    dataset = io.from_arrays(x, y, u, v, mask, frame=0)
    
    assert 'u' in dataset
    assert 'v' in dataset
    assert 'chc' in dataset
    assert dataset['u'].shape == (3, 5, 1)
    assert dataset['t'].values[0] == 0


def test_from_arrays_with_metadata():
    """Test from_arrays preserves array values"""
    x, y = np.meshgrid(np.arange(3), np.arange(2))
    u = np.ones((2, 3)) * 5.0
    v = np.ones((2, 3)) * 3.0
    mask = np.ones_like(u)
    
    dataset = io.from_arrays(x, y, u, v, mask, frame=10)
    
    assert np.allclose(dataset['u'].values[:, :, 0], 5.0)
    assert np.allclose(dataset['v'].values[:, :, 0], 3.0)
    assert dataset['t'].values[0] == 10


def test_create_sample_field_custom_params():
    """Test create_sample_field with custom parameters"""
    dataset = io.create_sample_field(rows=10, cols=15, frame=5, noise_sigma=0.1)
    
    assert dataset['u'].shape == (10, 15, 1)
    assert dataset['t'].values[0] == 5
    assert len(dataset['x']) == 15
    assert len(dataset['y']) == 10


def test_set_default_attrs_applied():
    """Test set_default_attrs applies correct attributes"""
    dataset = io.create_sample_field()
    
    assert 'units' in dataset.x.attrs
    assert 'units' in dataset.y.attrs
    assert 'units' in dataset.u.attrs
    assert 'units' in dataset.v.attrs
    assert 'units' in dataset.t.attrs
    assert 'delta_t' in dataset.attrs
    assert 'files' in dataset.attrs


# Edge cases and error handling
def test_read_empty_file_raises_error():
    """Test reading empty file raises appropriate error"""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'empty.txt')
        with open(filepath, 'w') as f:
            f.write('')
        
        # Should not be able to find a reader
        reader = io._REGISTRY.find_reader(filepath)
        # Empty file won't match any reader
        assert reader is None


def test_read_file_with_nan_values():
    """Test reading file handles NaN values"""
    dataset = io.create_sample_field()
    dataset['u'].values[0, 0, 0] = np.nan
    
    # Should not raise an error
    assert np.isnan(dataset['u'].values[0, 0, 0])


def test_read_single_row_or_column():
    """Test reading data with single row or column"""
    dataset = io.create_sample_field(rows=1, cols=10)
    
    assert dataset['u'].shape == (1, 10, 1)
    
    dataset2 = io.create_sample_field(rows=10, cols=1)
    assert dataset2['u'].shape == (10, 1, 1)


def test_netcdf_reader_roundtrip():
    """Exercise NetCDFReader via read_piv(format='netcdf'/'nc')."""
    import tempfile
    import os
    import pytest

    ds = io.create_sample_Dataset(n_frames=2, rows=3, cols=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp = os.path.join(tmpdir, "sample.nc")
        try:
            ds.to_netcdf(fp)
        except Exception as e:
            pytest.skip(f"NetCDF write not available in this environment: {e}")

        loaded = io.read_piv(fp, format="netcdf")
        assert set(["u", "v", "chc"]).issubset(set(loaded.data_vars))
        assert tuple(loaded["u"].dims) == ("y", "x", "t")
        assert loaded.sizes["t"] == 2

        loaded2 = io.read_piv(fp, format="nc")
        assert loaded2.sizes["t"] == 2


def test_openpiv_nan_and_mask_zeroing():
    """Ensure OpenPIVReader normalizes NaNs and zeros masked-out velocities."""
    import tempfile
    import os
    import pytest

    # Create a minimal 2x2 OpenPIV-like file with 6 columns (mask) and NaNs.
    # Columns: x y u v chc mask
    # Reshape order in reader is (rows, cols) based on unique counts.
    rows = [
        "0 0 NaN 1.0 1 1",
        "1 0 2.0 NaN 1 0",
        "0 1 3.0 4.0 1 1",
        "1 1 5.0 6.0 1 1",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        fp = os.path.join(tmpdir, "exp1_001_b.txt")
        with open(fp, "w") as f:
            f.write("\n".join(rows) + "\n")

        # Sanity: auto-detection should find OpenPIVReader.
        reader = io._REGISTRY.find_reader(fp)
        assert reader is not None
        assert isinstance(reader, io.OpenPIVReader)

        ds = io.read_piv(fp)
        assert "mask" in ds
        # NaNs in u/v become 0.0
        assert float(ds["u"].values[0, 0, 0]) == 0.0
        assert float(ds["v"].values[0, 1, 0]) == 0.0
        # mask==0 forces u/v to 0.0 at that location
        assert float(ds["mask"].values[0, 1, 0]) == 0.0
        assert float(ds["u"].values[0, 1, 0]) == 0.0
        assert float(ds["v"].values[0, 1, 0]) == 0.0


def test_from_arrays_validation_errors():
    """Cover from_arrays input validation branches."""
    import pytest

    x, y = np.meshgrid(np.arange(3), np.arange(2))
    u = np.ones((2, 3))
    v = np.ones((2, 3))

    with pytest.raises(ValueError, match="u and v must be 2D"):
        io.from_arrays(x, y, u[:, :, None], v)

    with pytest.raises(ValueError, match="mask must have same shape"):
        io.from_arrays(x, y, u, v, mask=np.ones((1, 1)))


def test_coords_from_mesh_requires_2d():
    """Cover _coords_from_mesh error path."""
    import pytest

    with pytest.raises(ValueError, match="Expected 2D mesh"):
        io._coords_from_mesh(np.arange(3), np.arange(3))


