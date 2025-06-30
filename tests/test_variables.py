import pytest
import numpy as np
import dask.array as da
from ad99py.variables import lapserate, bouyancy_freq_squared, density
from ad99py.constants import GRAV, R_DRY, C_P,BFLIM


class TestLapserate:
    """Test cases for the lapserate function."""
    
    def test_lapserate_numpy_linear(self):
        """Test lapserate with numpy arrays for linear temperature profile."""
        T = np.array([300.0, 295.0, 290.0, 285.0])
        z = np.array([0.0, 1000.0, 2000.0, 3000.0])
        
        result = lapserate(T, z)
        expected = np.gradient(T, axis=-1) / np.gradient(z, axis=-1)
        
        np.testing.assert_array_almost_equal(result, expected)
        assert result.shape == T.shape
    
    def test_lapserate_numpy_2d(self):
        """Test lapserate with 2D numpy arrays."""
        T = np.array([[300.0, 295.0, 290.0], [310.0, 305.0, 300.0]])
        z = np.array([[0.0, 1000.0, 2000.0], [0.0, 1000.0, 2000.0]])
        
        result = lapserate(T, z)
        expected = np.gradient(T, axis=-1) / np.gradient(z, axis=-1)
        
        np.testing.assert_array_almost_equal(result, expected)
        assert result.shape == T.shape
    
    def test_lapserate_dask_array(self):
        """Test lapserate with dask arrays."""
        T_np = np.array([300.0, 295.0, 290.0, 285.0])
        z_np = np.array([0.0, 1000.0, 2000.0, 3000.0])
        
        T_da = da.from_array(T_np, chunks=2)
        z_da = da.from_array(z_np, chunks=2)
        
        result = lapserate(T_da, z_da)
        expected = da.gradient(T_da, axis=-1) / da.gradient(z_da, axis=-1)
        
        np.testing.assert_array_almost_equal(result.compute(), expected.compute())
        assert isinstance(result, da.Array)
    
    def test_lapserate_mixed_inputs(self):
        """Test lapserate with mixed numpy and dask inputs."""
        T_np = np.array([300.0, 295.0, 290.0, 285.0])
        z_da = da.from_array(np.array([0.0, 1000.0, 2000.0, 3000.0]), chunks=2)
        
        result = lapserate(T_np, z_da)
        expected = da.gradient(T_np, axis=-1) / da.gradient(z_da, axis=-1)
        
        np.testing.assert_array_almost_equal(result.compute(), expected.compute())
        assert isinstance(result, da.Array)


class TestBouyancyFreqSquared:
    """Test cases for the bouyancy_freq_squared function."""
    
    def test_bouyancy_freq_squared_numpy_basic(self):
        """Test bouyancy_freq_squared with numpy arrays."""
        T = np.array([300.0, 295.0, 290.0, 285.0])
        z = np.array([0.0, 1000.0, 2000.0, 3000.0])
        bflim = 5e-3
        
        result = bouyancy_freq_squared(T, z, bflim)
        
        # Calculate expected result manually
        lr = lapserate(T, z)
        Ns2unfilter = (GRAV/T) * (lr + GRAV/C_P)
        expected = np.where(Ns2unfilter < bflim**2, bflim**2, Ns2unfilter)
        
        np.testing.assert_array_almost_equal(result, expected)
        assert result.shape == T.shape
    
    def test_bouyancy_freq_squared_with_limit(self):
        """Test that the function properly applies the bflim constraint."""
        # Create a scenario where some values will be below the limit
        T = np.array([300.0, 300.1, 300.2, 300.3])  # Very small temperature gradient
        z = np.array([0.0, 1000.0, 2000.0, 3000.0])
        bflim = 1e-2  # Larger limit
        
        result = bouyancy_freq_squared(T, z, bflim)
        
        # All values should be at least bflim**2
        assert np.all(result >= bflim**2)
    
    def test_bouyancy_freq_squared_dask_array(self):
        """Test bouyancy_freq_squared with dask arrays."""
        T_np = np.array([300.0, 295.0, 290.0, 285.0])
        z_np = np.array([0.0, 1000.0, 2000.0, 3000.0])
        
        T_da = da.from_array(T_np, chunks=2)
        z_da = da.from_array(z_np, chunks=2)
        
        result = bouyancy_freq_squared(T_da, z_da)
        
        # Compare with numpy version
        expected = bouyancy_freq_squared(T_np, z_np)
        
        np.testing.assert_array_almost_equal(result.compute(), expected)
        assert isinstance(result, da.Array)
    
    def test_bouyancy_freq_squared_default_bflim(self):
        """Test bouyancy_freq_squared with default bflim value."""
        T = np.array([300.0, 295.0, 290.0, 285.0])
        z = np.array([0.0, 1000.0, 2000.0, 3000.0])
        
        result = bouyancy_freq_squared(T, z)  # Use default bflim
        
        # Should use default bflim = 5e-3
        expected = bouyancy_freq_squared(T, z, BFLIM)
        
        np.testing.assert_array_almost_equal(result, expected)


class TestDensity:
    """Test cases for the density function."""
    
    def test_density_numpy_hectopascal_true(self):
        """Test density calculation with hectopascal=True (default)."""
        T = np.array([300.0, 295.0, 290.0])
        p = np.array([1000.0, 850.0, 700.0])  # hPa
        
        result = density(T, p, hectopascal=True)
        
        # Expected calculation: pbrd = 100*p, then pbrd/(R_DRY*T)
        pbrd = 100 * p
        expected = pbrd / (R_DRY * T)
        
        np.testing.assert_array_almost_equal(result, expected)
        assert result.shape == T.shape
    
    def test_density_numpy_hectopascal_false(self):
        """Test density calculation with hectopascal=False."""
        T = np.array([300.0, 295.0, 290.0])
        p = np.array([100000.0, 85000.0, 70000.0])  # Pa
        
        result = density(T, p, hectopascal=False)
        
        # Expected calculation: pbrd = 1*p, then pbrd/(R_DRY*T)
        pbrd = 1 * p
        expected = pbrd / (R_DRY * T)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_density_dask_array(self):
        """Test density calculation with dask arrays."""
        T_np = np.array([300.0, 295.0, 290.0])
        p_np = np.array([1000.0, 850.0, 700.0])
        
        T_da = da.from_array(T_np, chunks=2)
        p_da = da.from_array(p_np, chunks=2)
        
        result = density(T_da, p_da)
        
        # Compare with numpy version
        expected = density(T_np, p_np)
        
        np.testing.assert_array_almost_equal(result.compute(), expected)
        assert isinstance(result, da.Array)
    
    def test_density_broadcasting(self):
        """Test density calculation with broadcasting."""
        T = np.array([[300.0, 295.0], [290.0, 285.0]])  # 2x2
        p = np.array([1000.0, 850.0])  # 1D array that should broadcast
        
        result = density(T, p)
        
        # Manual calculation with broadcasting
        pbrd = np.broadcast_to(100 * p, T.shape)
        expected = pbrd / (R_DRY * T)
        
        np.testing.assert_array_almost_equal(result, expected)
        assert result.shape == T.shape
    
    def test_density_dask_broadcasting(self):
        """Test density calculation with dask arrays and broadcasting."""
        T_np = np.array([[300.0, 295.0], [290.0, 285.0]])
        p_np = np.array([1000.0, 850.0])
        
        T_da = da.from_array(T_np, chunks=(1, 2))
        p_da = da.from_array(p_np, chunks=2)
        
        result = density(T_da, p_da)
        
        # Compare with numpy version
        expected = density(T_np, p_np)
        
        np.testing.assert_array_almost_equal(result.compute(), expected)
        assert isinstance(result, da.Array)
    
    def test_density_edge_cases(self):
        """Test density calculation with edge cases."""
        # Test with very small temperature values
        T = np.array([1.0, 2.0, 3.0])
        p = np.array([1000.0, 850.0, 700.0])
        
        result = density(T, p)
        
        # Should not raise any errors and produce finite results
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)  # Density should be positive


class TestIntegration:
    """Integration tests for functions working together."""
    
    def test_realistic_atmospheric_profile(self):
        """Test with realistic atmospheric temperature and height profiles."""
        # Typical atmospheric profile (temperature decreases with height)
        heights = np.array([0, 500, 1000, 1500, 2000, 2500, 3000])  # meters
        temperatures = np.array([288, 285, 282, 279, 276, 273, 270])  # Kelvin
        pressures = np.array([1013, 955, 899, 846, 795, 747, 701])  # hPa
        
        # All functions should work without errors
        lr = lapserate(temperatures, heights)
        n2 = bouyancy_freq_squared(temperatures, heights)
        rho = density(temperatures, pressures)
        
        # Basic sanity checks
        assert lr.shape == temperatures.shape
        assert n2.shape == temperatures.shape
        assert rho.shape == temperatures.shape
        assert np.all(np.isfinite(lr))
        assert np.all(np.isfinite(n2))
        assert np.all(np.isfinite(rho))
        assert np.all(rho > 0)  # Density should be positive
        assert np.all(n2 > 0)   # Buoyancy frequency squared should be positive