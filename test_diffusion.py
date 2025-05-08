"""Tests for thr diffusion model module."""
import pytest
import math
from diffusion import calculate_stable_time_step
def test_time_step_is_float():
    time_step=calculate_stable_time_step(dx=1,diffusivity=1)
    assert isinstance(time_step,float)

def test_time_step_with_zero_spacing():
    dt=calculate_stable_time_step(dx=0.0,diffusivity=1)
    assert dt==pytest.approx(0.0)
    assert math.isclose(dt,0.0)
    