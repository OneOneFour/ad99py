import pytest
from ad99py.constants import GRAV, R_DRY, GAMMA, C_P, BFLIM, C_V

def test_gravity():
    assert GRAV == pytest.approx(9.81), "Incorrect gravitational acceleration"

def test_r_dry():
    assert R_DRY == pytest.approx(287.04), "Incorrect gas constant for dry air"

def test_gamma():
    assert GAMMA == pytest.approx(1.4), "Incorrect adiabatic index"

def test_c_p_definition():
    # C_P = R / (1 - 1/GAMMA)
    expected_cp = R_DRY / (1 - 1 / GAMMA)
    assert C_P == pytest.approx(expected_cp), "Incorrect C_P calculation"

def test_c_v_definition():
    # C_V = C_P - R
    expected_cv = C_P - R_DRY
    assert C_V == pytest.approx(expected_cv), "Incorrect C_V calculation"

def test_gamma_consistency():
    # Invert: gamma = C_P / C_V
    inferred_gamma = C_P / C_V
    assert inferred_gamma == pytest.approx(GAMMA, rel=1e-6), "Gamma not consistent with C_P/C_V"

def test_bflim():
    assert BFLIM == pytest.approx(5e-3), "Incorrect BFLIM value"