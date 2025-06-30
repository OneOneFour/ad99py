import pytest
import numpy as np
import xarray as xr
import os
import tempfile
from unittest.mock import patch, Mock
from ad99py.masks import default_path, load_mask, mask_dataset, DEFAULT_MASK_NAME, DEFAULT_MASK_DIR


class TestDefaultPath:
    """Test cases for the default_path function."""
    
    def test_default_path_with_defaults(self):
        """Test default_path with default parameters."""
        result = default_path()
        expected = os.path.join(DEFAULT_MASK_DIR, DEFAULT_MASK_NAME)
        assert result == expected
        assert result == "data/loon_masks.nc"
    
    def test_default_path_custom_dir(self):
        """Test default_path with custom directory."""
        custom_dir = "custom_data"
        result = default_path(dir=custom_dir)
        expected = os.path.join(custom_dir, DEFAULT_MASK_NAME)
        assert result == expected
    
    def test_default_path_custom_name(self):
        """Test default_path with custom filename."""
        custom_name = "custom_mask.nc"
        result = default_path(name=custom_name)
        expected = os.path.join(DEFAULT_MASK_DIR, custom_name)
        assert result == expected
    
    def test_default_path_custom_both(self):
        """Test default_path with both custom directory and filename."""
        custom_dir = "custom_data"
        custom_name = "custom_mask.nc"
        result = default_path(dir=custom_dir, name=custom_name)
        expected = os.path.join(custom_dir, custom_name)
        assert result == expected


class TestLoadMask:
    """Test cases for the load_mask function."""
    
    def create_mock_mask_dataset(self, lon_range=(-180, 180)):
        """Helper method to create a mock mask dataset."""
        lon_min, lon_max = lon_range
        lons = np.linspace(lon_min, lon_max, 10)
        lats = np.linspace(-90, 90, 8)
        
        # Create mask data
        mask_data = np.ones((len(lats), len(lons)))
        
        return xr.Dataset({
            'mask1': (('lat', 'lon'), mask_data),
            'mask2': (('lat', 'lon'), mask_data * 0.5)
        }, coords={
            'lat': lats,
            'lon': lons
        })
    
    @patch('xarray.open_dataset')
    def test_load_mask_default_path(self, mock_open_dataset):
        """Test load_mask with default path."""
        mock_dataset = self.create_mock_mask_dataset()
        mock_open_dataset.return_value = mock_dataset
        
        result = load_mask()
        
        # Check that open_dataset was called with the default path
        mock_open_dataset.assert_called_once_with(default_path())
        
        # Check recentering happened (longitude values should be 0-360)
        assert np.all(result.lon >= 0)
        assert np.all(result.lon <= 360)
    
    @patch('xarray.open_dataset')
    def test_load_mask_custom_path(self, mock_open_dataset):
        """Test load_mask with custom path."""
        mock_dataset = self.create_mock_mask_dataset()
        mock_open_dataset.return_value = mock_dataset
        custom_path = "/custom/path/mask.nc"
        
        result = load_mask(path=custom_path)
        
        mock_open_dataset.assert_called_once_with(custom_path)
    
    @patch('xarray.open_dataset')
    def test_load_mask_no_recentering(self, mock_open_dataset):
        """Test load_mask without recentering."""
        mock_dataset = self.create_mock_mask_dataset(lon_range=(-180, 180))
        mock_open_dataset.return_value = mock_dataset
        
        result = load_mask(recentering=False)
        
        # Longitude values should remain unchanged
        np.testing.assert_array_equal(result.lon.values, mock_dataset.lon.values)
    
    @patch('xarray.open_dataset')
    def test_load_mask_recentering_negative_lons(self, mock_open_dataset):
        """Test load_mask recentering with negative longitudes."""
        mock_dataset = self.create_mock_mask_dataset(lon_range=(-180, 0))
        mock_open_dataset.return_value = mock_dataset
        
        result = load_mask(recentering=True)
        
        # All longitude values should be positive after recentering
        assert np.all(result.lon >= 0)
        assert np.all(result.lon <= 360)


class TestMaskDataset:
    """Test cases for the mask_dataset function."""
    
    def create_mock_data_dataset_latlon(self, has_data=True):
        """Helper to create mock dataset with lat/lon coordinates."""
        lons = np.linspace(0, 360, 12)
        lats = np.linspace(-90, 90, 9)
        
        if has_data:
            temp_data = 273.15 + np.random.randn(len(lats), len(lons))
            pressure_data = 1013.25 + np.random.randn(len(lats), len(lons)) * 50
        else:
            temp_data = np.full((len(lats), len(lons)), np.nan)
            pressure_data = np.full((len(lats), len(lons)), np.nan)
        
        return xr.Dataset({
            'temperature': (('lat', 'lon'), temp_data),
            'pressure': (('lat', 'lon'), pressure_data)
        }, coords={
            'lat': lats,
            'lon': lons
        })
    
    def create_mock_data_dataset_latitude_longitude(self, has_data=True):
        """Helper to create mock dataset with latitude/longitude coordinates."""
        lons = np.linspace(0, 360, 12)
        lats = np.linspace(-90, 90, 9)
        
        if has_data:
            temp_data = 273.15 + np.random.randn(len(lats), len(lons))
            pressure_data = 1013.25 + np.random.randn(len(lats), len(lons)) * 50
        else:
            temp_data = np.full((len(lats), len(lons)), np.nan)
            pressure_data = np.full((len(lats), len(lons)), np.nan)
        
        return xr.Dataset({
            'temperature': (('latitude', 'longitude'), temp_data),
            'pressure': (('latitude', 'longitude'), pressure_data)
        }, coords={
            'latitude': lats,
            'longitude': lons
        })
    
    def create_mock_mask_dataset(self):
        """Helper to create mock mask dataset."""
        lons = np.linspace(0, 360, 10)
        lats = np.linspace(-90, 90, 8)
        
        # Create mask with some regions masked out
        mask1_data = np.ones((len(lats), len(lons)))
        mask1_data[:, :3] = 0  # Mask out some western regions
        
        mask2_data = np.ones((len(lats), len(lons)))
        mask2_data[-2:, :] = 0  # Mask out some southern regions
        
        return xr.Dataset({
            'ocean_mask': (('lat', 'lon'), mask1_data),
            'land_mask': (('lat', 'lon'), mask2_data)
        }, coords={
            'lat': lats,
            'lon': lons
        })
    
    @patch('ad99py.masks.load_mask')
    def test_mask_dataset_with_latlon_coords(self, mock_load_mask):
        """Test mask_dataset with lat/lon coordinate names."""
        mock_mask = self.create_mock_mask_dataset()
        mock_load_mask.return_value = mock_mask
        
        data_ds = self.create_mock_data_dataset_latlon()
        
        result = mask_dataset(data_ds)
        
        # Check that the result has 'points' dimension instead of lat/lon
        assert 'points' in result.dims
        assert 'lat' not in result.dims
        assert 'lon' not in result.dims
        
        # Check that data variables are preserved
        assert 'temperature' in result.data_vars
        assert 'pressure' in result.data_vars
        
        # Check that points coordinate still exist
        assert 'lat' in result.coords
        assert 'lon' in result.coords
    
    @patch('ad99py.masks.load_mask')
    def test_mask_dataset_with_latitude_longitude_coords(self, mock_load_mask):
        """Test mask_dataset with latitude/longitude coordinate names."""
        mock_mask = self.create_mock_mask_dataset()
        mock_load_mask.return_value = mock_mask
        
        data_ds = self.create_mock_data_dataset_latitude_longitude()
        
        result = mask_dataset(data_ds)
        
        # Check that the result has 'points' dimension instead of latitude/longitude
        assert 'points' in result.dims
        assert 'latitude' not in result.dims
        assert 'longitude' not in result.dims
        
        # Check that data variables are preserved
        assert 'temperature' in result.data_vars
        assert 'pressure' in result.data_vars
        
        # Check that points coordinate still exist
        assert 'latitude' in result.coords
        assert 'longitude' in result.coords
    
    def test_mask_dataset_with_provided_mask(self):
        """Test mask_dataset with explicitly provided mask."""
        data_ds = self.create_mock_data_dataset_latlon()
        mask_ds = self.create_mock_mask_dataset()
        
        result = mask_dataset(data_ds, mask=mask_ds)
        
        # Check that the result has 'points' dimension
        assert 'points' in result.dims
        assert 'lat' not in result.dims
        assert 'lon' not in result.dims
    
    def test_mask_dataset_invalid_coordinates(self):
        """Test mask_dataset with invalid coordinate names."""
        # Create dataset with neither lat/lon nor latitude/longitude
        invalid_ds = xr.Dataset({
            'temperature': (('x', 'y'), np.random.randn(5, 5))
        }, coords={
            'x': np.arange(5),
            'y': np.arange(5)
        })
        
        mask_ds = self.create_mock_mask_dataset()
        
        # This should raise an error when trying to interpolate
        with pytest.raises((KeyError, ValueError)):
            mask_dataset(invalid_ds, mask=mask_ds)
    
    @patch('ad99py.masks.load_mask')
    def test_mask_dataset_removes_all_nan_points(self, mock_load_mask):
        """Test that mask_dataset properly removes points with all NaN values."""
        # Create mask that allows some data through
        mock_mask = self.create_mock_mask_dataset()
        mock_load_mask.return_value = mock_mask
        
        # Create dataset with some NaN regions
        data_ds = self.create_mock_data_dataset_latlon(has_data=True)
        # Add some NaN values
        data_ds['temperature'].values[0, :3] = np.nan
        data_ds['pressure'].values[0, :3] = np.nan
        
        result = mask_dataset(data_ds)
        
        # Check that result has fewer points than original grid
        original_size = data_ds.lat.size * data_ds.lon.size
        assert result.points.size < original_size
        
        # Check that no points have all NaN values
        for var in result.data_vars:
            assert not np.all(np.isnan(result[var].values))
    
    def test_mask_dataset_coordinate_handling(self):
        """Test that mask_dataset properly handles coordinate renaming."""
        data_ds = self.create_mock_data_dataset_latitude_longitude()
        mask_ds = self.create_mock_mask_dataset()  # This has lat/lon coords
        
        result = mask_dataset(data_ds, mask=mask_ds)
        
        # The function should handle the coordinate name mismatch internally
        assert 'points' in result.dims
        assert 'temperature' in result.data_vars
        assert 'pressure' in result.data_vars


class TestIntegration:
    """Integration tests for the masks module."""
    
    def create_realistic_test_data(self):
        """Create realistic test datasets for integration testing."""
        # Create a realistic geographic grid
        lons = np.linspace(-180, 179, 72)  # 5-degree resolution
        lats = np.linspace(-90, 90, 37)    # 5-degree resolution
        
        # Create realistic atmospheric data
        temp_data = 273.15 + 20 * np.sin(np.radians(lats))[:, None] + np.random.randn(len(lats), len(lons)) * 5
        pressure_data = 1013.25 * np.exp(-lats[:, None]**2 / (2 * 30**2)) + np.random.randn(len(lats), len(lons)) * 10
        
        data_ds = xr.Dataset({
            'temperature': (('lat', 'lon'), temp_data),
            'pressure': (('lat', 'lon'), pressure_data)
        }, coords={
            'lat': lats,
            'lon': lons
        })
        
        # Create a realistic mask (e.g., ocean mask)
        mask_lons = np.linspace(-180, 179, 36)
        mask_lats = np.linspace(-90, 90, 19)
        
        # Simple ocean mask: mask out some "continental" regions
        ocean_mask = np.ones((len(mask_lats), len(mask_lons)))
        ocean_mask[8:12, 15:25] = 0  # "North America"
        ocean_mask[5:10, 30:35] = 0  # "Europe"
        
        mask_ds = xr.Dataset({
            'ocean': (('lat', 'lon'), ocean_mask)
        }, coords={
            'lat': mask_lats,
            'lon': mask_lons
        })
        
        return data_ds, mask_ds
    
    def test_full_workflow_latlon(self):
        """Test the complete workflow with lat/lon coordinates."""
        data_ds, mask_ds = self.create_realistic_test_data()
        
        result = mask_dataset(data_ds, mask=mask_ds)
        
        # Comprehensive checks
        assert isinstance(result, xr.Dataset)
        assert 'points' in result.dims
        assert 'lat' not in result.dims
        assert 'lon' not in result.dims
        assert 'temperature' in result.data_vars
        assert 'pressure' in result.data_vars
        
        # Check that masking actually removed some points
        original_size = data_ds.lat.size * data_ds.lon.size
        assert result.points.size < original_size
        
        # Check that remaining data is finite
        assert np.all(np.isfinite(result.temperature.values))
        assert np.all(np.isfinite(result.pressure.values))
    
    def test_full_workflow_latitude_longitude(self):
        """Test the complete workflow with latitude/longitude coordinates."""
        data_ds, mask_ds = self.create_realistic_test_data()
        
        # Rename coordinates to latitude/longitude
        data_ds = data_ds.rename({'lat': 'latitude', 'lon': 'longitude'})
        
        result = mask_dataset(data_ds, mask=mask_ds)
        
        # Same checks as above
        assert isinstance(result, xr.Dataset)
        assert 'points' in result.dims
        assert 'latitude' not in result.dims
        assert 'longitude' not in result.dims
        assert 'temperature' in result.data_vars
        assert 'pressure' in result.data_vars
    
    @patch('ad99py.masks.load_mask')
    def test_error_handling_file_not_found(self, mock_load_mask):
        """Test error handling when mask file is not found."""
        mock_load_mask.side_effect = FileNotFoundError("Mask file not found")
        
        data_ds, _ = self.create_realistic_test_data()
        
        with pytest.raises(FileNotFoundError):
            mask_dataset(data_ds)
    