import pytest
import numpy as np
from ad99py.sources import (
    make_source_spectrum,
    uniform_source,
    gaussian_source,
    convective_source
)


class TestSourceFunctions:
    """Simple tests for source spectrum functions."""
    
    def setup_method(self):
        """Set up common test parameters."""
        self.c = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # phase speeds
        self.c0 = 30.0  # reference phase speed
        self.cw = 10.0  # width parameter
        self.Bm = 1.0   # magnitude parameter
    
    def test_uniform_source(self):
        """Test uniform source returns constant Bm."""
        result = uniform_source(self.c, self.c0, self.cw, self.Bm)
        expected = self.Bm
        
        assert result == expected
        # Should work with scalar inputs too
        assert uniform_source(25.0, self.c0, self.cw, self.Bm) == self.Bm
    
    def test_gaussian_source_basic(self):
        """Test Gaussian source basic properties."""
        result = gaussian_source(self.c, self.c0, self.cw, self.Bm)
        
        # Should be array of same shape as c
        assert result.shape == self.c.shape
        
        # All values should be positive
        assert np.all(result >= 0)
        
        # Maximum should be at c0
        c_fine = np.linspace(10, 50, 100)
        result_fine = gaussian_source(c_fine, self.c0, self.cw, self.Bm)
        max_idx = np.argmax(result_fine)
        assert c_fine[max_idx] == pytest.approx(self.c0, abs=0.5)
        
        # Value at c0 should be Bm
        assert gaussian_source(self.c0, self.c0, self.cw, self.Bm) == pytest.approx(self.Bm)
    
    def test_gaussian_source_symmetry(self):
        """Test Gaussian source is symmetric around c0."""
        delta = 5.0
        left = gaussian_source(self.c0 - delta, self.c0, self.cw, self.Bm)
        right = gaussian_source(self.c0 + delta, self.c0, self.cw, self.Bm)
        
        assert left == pytest.approx(right, rel=1e-10)
    
    def test_convective_source_basic(self):
        """Test convective source basic properties."""
        result = convective_source(self.c, self.c0, self.cw, self.Bm)
        
        # Should be array of same shape as c
        assert result.shape == self.c.shape
        
        # Value at c0 should be 0 (due to (c-c0)/cw factor)
        assert convective_source(self.c0, self.c0, self.cw, self.Bm) == pytest.approx(0.0)
    
    def test_convective_source_asymmetry(self):
        """Test convective source is asymmetric around c0."""
        delta = 5.0
        left = convective_source(self.c0 - delta, self.c0, self.cw, self.Bm)
        right = convective_source(self.c0 + delta, self.c0, self.cw, self.Bm)
        
        # Should have opposite signs
        assert np.sign(left) == -np.sign(right)
        assert np.abs(left) == pytest.approx(np.abs(right))


class TestMakeSourceSpectrum:
    """Test the source spectrum factory function."""
    
    def test_make_source_spectrum_gaussian(self):
        """Test make_source_spectrum with Gaussian source."""
        cw, Bm = 10.0, 1.0
        source_func = make_source_spectrum(gaussian_source, cw, Bm)
        
        # Function should take only c and c0 as arguments
        c, c0 = 25.0, 30.0
        result = source_func(c, c0)
        
        # Should give same result as calling gaussian_source directly
        expected = gaussian_source(c, c0, cw, Bm)
        assert result == pytest.approx(expected)
    
    def test_make_source_spectrum_uniform(self):
        """Test make_source_spectrum with uniform source."""
        cw, Bm = 5.0, 2.0
        source_func = make_source_spectrum(uniform_source, cw, Bm)
        
        result = source_func(25.0, 30.0)
        expected = uniform_source(25.0, 30.0, cw, Bm)
        
        assert result == pytest.approx(expected)
        assert result == pytest.approx(Bm)  # Should just return Bm
    
    def test_make_source_spectrum_convective(self):
        """Test make_source_spectrum with convective source."""
        cw, Bm = 8.0, 1.5
        source_func = make_source_spectrum(convective_source, cw, Bm)
        
        c_array = np.array([20.0, 30.0, 40.0])
        c0 = 30.0
        
        result = source_func(c_array, c0)
        expected = convective_source(c_array, c0, cw, Bm)
        
        np.testing.assert_array_almost_equal(result, expected)


class TestSourceParameterSensitivity:
    """Test how sources respond to parameter changes."""
    
    def test_gaussian_width_parameter(self):
        """Test Gaussian source width parameter effect."""
        c = np.linspace(20, 40, 21)
        c0 = 30.0
        Bm = 1.0
        
        # Narrow Gaussian
        cw_narrow = 2.0
        result_narrow = gaussian_source(c, c0, cw_narrow, Bm)
        
        # Wide Gaussian  
        cw_wide = 20.0
        result_wide = gaussian_source(c, c0, cw_wide, Bm)
        
        # Narrow should have higher peak and faster decay
        peak_narrow = np.max(result_narrow)
        peak_wide = np.max(result_wide)
        
        # Both should peak at Bm
        assert peak_narrow == pytest.approx(Bm)
        assert peak_wide == pytest.approx(Bm)
        
        # Narrow should decay faster away from peak
        edge_idx = 0  # Far from c0=30
        assert result_narrow[edge_idx] < result_wide[edge_idx]
    
    def test_magnitude_scaling(self):
        """Test that magnitude parameter scales all sources correctly."""
        c, c0, cw = 25.0, 30.0, 10.0
        Bm_small, Bm_large = 0.5, 2.0
        
        # Test all source types
        sources = [uniform_source, gaussian_source, convective_source]
        
        for source_func in sources:
            small = source_func(c, c0, cw, Bm_small)
            large = source_func(c, c0, cw, Bm_large)
            
            # Should scale proportionally
            if source_func == uniform_source:
                assert large == pytest.approx(4 * small)  # 2.0/0.5 = 4
            else:
                assert large == pytest.approx(4 * small)