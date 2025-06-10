import pytest 
import numpy as np
from ad99py.ad99 import AlexanderDunkerton1999

def test_default_ad99_initialization():
    ad99 = AlexanderDunkerton1999()
    assert ad99 is not None, "Failed to initialize AlexanderDunkerton1999 class"

    assert ad99.source_level_height == 9000
    assert ad99.Bm == 0.4
    assert ad99.cw == 35
    assert ad99.damp_level_height is None
    assert ad99.cmax == 99.6
    assert ad99.dc == 1.2
    assert ad99.base_wavelength == 300e3
    assert ad99.force_intermittency is None
    assert not ad99.no_alpha
    assert not ad99.exclude_unbroken

def test_ad99_force_intermittency():
    ad99 = AlexanderDunkerton1999(force_intermittency=1.0)
    rho = 1.0
    assert ad99.intermittency(rho) == pytest.approx(1.0)

def test_ad99_exclude_topwaves():
    ad99 = AlexanderDunkerton1999(exclude_unbroken=True)
    assert ad99.exclude_topwaves

def test_ad99_get_source_level_height():
    z = np.arange(0,20000,1000)
    ad99 = AlexanderDunkerton1999(source_level_height=10e3)
    assert ad99.get_source_level(z) == 10 

def test_ad99_null_drag_case():
    ad99 = AlexanderDunkerton1999()
    z = np.arange(0, 20000, 1000,dtype=float)
    u = np.zeros_like(z)
    N = np.ones_like(u)
    rho = np.ones_like(u)
    gwd = ad99.gwd(u,N,z,rho)
    assert np.allclose(gwd,0.0), "GWD should be zero for null drag case"

def test_ad99_alt_momentum_flux_abs_null_broken():
    ad99 = AlexanderDunkerton1999(exclude_unbroken=False)
    z = np.arange(0, 20000, 1000,dtype=float)
    u = np.zeros_like(z)
    N = np.ones_like(u)
    rho = np.ones_like(u)
    mflux = ad99.momentum_flux_abs(u, N, z, rho)
    assert not np.allclose(mflux, 0.0), "Momentum flux should be not zero for null drag case with breaking waves"

def test_ad99_alt_momentum_flux_abs_null_unbroken():
    ad99 = AlexanderDunkerton1999(exclude_unbroken=True)
    z = np.arange(0, 20000, 1000,dtype=float)
    u = np.zeros_like(z)
    N = np.ones_like(u)
    rho = np.ones_like(u)
    mflux = ad99.momentum_flux_abs(u, N, z, rho)
    assert np.allclose(mflux, 0.0), "Momentum flux should be zero for null drag case"

def test_ad99_momentum_flux_abs_null_unbroken():
    ad99 = AlexanderDunkerton1999(exclude_unbroken=True)
    z = np.arange(0, 20000, 1000,dtype=float)
    u = np.zeros_like(z)
    N = np.ones_like(u)
    rho = np.ones_like(u)
    ntv,ptv = ad99.momentum_flux_neg_ptv(u, N, z, rho)
    assert np.allclose(ntv, 0.0), "Momentum flux should be zero for null drag case"
    assert np.allclose(ptv, 0.0), "Momentum flux should be zero for null drag case"


def test_ad99_momentum_flux_null_broken():
    ad99 = AlexanderDunkerton1999(exclude_unbroken=False)
    z = np.arange(0, 20000, 1000,dtype=float)
    u = np.zeros_like(z)
    N = np.ones_like(u)
    rho = np.ones_like(u)
    ntv,ptv = ad99.momentum_flux_neg_ptv(u, N, z, rho)
    assert not np.allclose(ntv, 0.0), "Momentum flux should be zero for null drag case"
    assert not np.allclose(ptv, 0.0), "Momentum flux should be zero for null drag case"
    net = ntv + ptv 
    abs = np.abs(ntv) + np.abs(ptv)
    assert not np.allclose(abs,0.0)
    assert np.allclose(net, 0.0), "Net momentum flux should be zero for null drag case"