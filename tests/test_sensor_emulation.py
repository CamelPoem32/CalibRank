import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calib_observability.lie_se3 import se3_exp
from calib_observability.sensor_emulation import (
    SE3CalibrationLaw,
    ScalarCalibrationLaw,
    emulate_time_varying_imu_calibration,
    warp_sensor_timestamps,
)


def _rz(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return T


def test_constant_laws_reproduce_original_imu_stream():
    timestamps = np.linspace(0.0, 1.0, 11)
    gyroscope = np.column_stack([timestamps, 0.2 * timestamps, -0.1 + timestamps])
    accelerometer = np.column_stack([np.sin(timestamps), np.cos(timestamps), np.ones_like(timestamps) * 9.81])
    T_ref = se3_exp(np.array([0.1, -0.05, 0.02, 0.3, -0.2, 0.1]))
    tau_ref = 0.07
    bias = np.array([0.01, -0.02, 0.03])

    result = emulate_time_varying_imu_calibration(
        sensor_timestamps=timestamps,
        gyroscope=gyroscope,
        accelerometer=accelerometer,
        T_B_I_reference=T_ref,
        tau_I_reference=tau_ref,
        T_B_I_law=SE3CalibrationLaw("constant", reference=T_ref),
        tau_I_law=ScalarCalibrationLaw("constant", value=tau_ref),
        bias_reference=bias,
        include_lever_arm_correction=True,
    )

    assert np.allclose(result.reference_timestamps, timestamps - tau_ref)
    assert np.allclose(result.sensor_timestamps, timestamps)
    assert np.allclose(result.gyroscope, gyroscope)
    assert np.allclose(result.accelerometer, accelerometer)


def test_linear_scalar_law_hits_endpoints():
    law = ScalarCalibrationLaw("linear", t_start=10.0, t_end=20.0, start=0.05, end=0.25)
    values = law([10.0, 15.0, 20.0])
    assert np.allclose(values, [0.05, 0.15, 0.25])


def test_linear_se3_law_hits_endpoints_and_stays_se3():
    T_start = se3_exp(np.array([0.01, 0.02, -0.03, 0.1, -0.2, 0.3]))
    T_end = T_start @ se3_exp(np.array([0.1, -0.05, 0.08, 0.2, 0.0, -0.1]))
    law = SE3CalibrationLaw("linear", t_start=0.0, t_end=2.0, start=T_start, end=T_end)
    transforms = law([0.0, 1.0, 2.0])

    assert np.allclose(transforms[0], T_start)
    assert np.allclose(transforms[-1], T_end)
    for transform in transforms:
        assert np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0])
        assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-10)


def test_known_gyro_vector_reorientation():
    timestamps = np.array([0.0, 0.5, 1.0])
    gyroscope = np.tile([1.0, 0.0, 0.0], (3, 1))
    accelerometer = np.zeros((3, 3))
    result = emulate_time_varying_imu_calibration(
        sensor_timestamps=timestamps,
        gyroscope=gyroscope,
        accelerometer=accelerometer,
        T_B_I_reference=np.eye(4),
        tau_I_reference=0.0,
        T_B_I_law=SE3CalibrationLaw("constant", reference=_rz(np.pi / 2.0)),
        tau_I_law=ScalarCalibrationLaw("constant", value=0.0),
        bias_reference=np.zeros(3),
    )

    assert np.allclose(result.gyroscope, np.tile([0.0, -1.0, 0.0], (3, 1)), atol=1e-12)


def test_timestamp_warp_uses_factor_convention():
    timestamps = np.array([1.0, 2.0, 3.0])
    law = ScalarCalibrationLaw("linear", t_start=0.5, t_end=2.5, start=0.1, end=0.3)
    reference_times, warped, tau_truth = warp_sensor_timestamps(timestamps, 0.5, law)

    assert np.allclose(reference_times, [0.5, 1.5, 2.5])
    assert np.allclose(tau_truth, [0.1, 0.2, 0.3])
    assert np.allclose(warped, reference_times + tau_truth)


def test_timestamp_warp_rejects_non_monotonic_output():
    timestamps = np.array([0.0, 1.0, 2.0])
    law = ScalarCalibrationLaw("piecewise", control_times=[0.0, 1.0, 2.0], control_values=[0.0, -2.0, -4.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        warp_sensor_timestamps(timestamps, 0.0, law)


def test_lever_arm_correction_identity_when_lever_arm_is_unchanged():
    timestamps = np.linspace(0.0, 2.0, 21)
    gyroscope = np.column_stack([0.1 + timestamps, -0.2 + 0.5 * timestamps, 0.05 * timestamps])
    accelerometer = np.column_stack([np.sin(timestamps), np.cos(timestamps), 9.81 + 0.1 * timestamps])
    T_ref = se3_exp(np.array([0.05, -0.02, 0.03, 0.4, -0.1, 0.2]))

    result = emulate_time_varying_imu_calibration(
        sensor_timestamps=timestamps,
        gyroscope=gyroscope,
        accelerometer=accelerometer,
        T_B_I_reference=T_ref,
        tau_I_reference=0.0,
        T_B_I_law=SE3CalibrationLaw("constant", reference=T_ref),
        tau_I_law=ScalarCalibrationLaw("constant", value=0.0),
        include_lever_arm_correction=True,
    )

    assert np.allclose(result.accelerometer, accelerometer)


def test_emulation_does_not_mutate_inputs():
    timestamps = np.linspace(0.0, 1.0, 5)
    gyroscope = np.ones((5, 3))
    accelerometer = np.ones((5, 3)) * 9.81
    timestamps_copy = timestamps.copy()
    gyroscope_copy = gyroscope.copy()
    accelerometer_copy = accelerometer.copy()

    emulate_time_varying_imu_calibration(
        sensor_timestamps=timestamps,
        gyroscope=gyroscope,
        accelerometer=accelerometer,
        T_B_I_reference=np.eye(4),
        tau_I_reference=0.0,
        T_B_I_law=SE3CalibrationLaw("linear", t_start=0.0, t_end=1.0, reference=np.eye(4), end_delta_xi=[0.1, 0, 0, 0, 0, 0]),
        tau_I_law=ScalarCalibrationLaw("constant", value=0.0),
    )

    assert np.array_equal(timestamps, timestamps_copy)
    assert np.array_equal(gyroscope, gyroscope_copy)
    assert np.array_equal(accelerometer, accelerometer_copy)


def test_piecewise_laws_are_piecewise_constant():
    scalar_law = ScalarCalibrationLaw("piecewise", control_times=[0.0, 1.0, 2.0], control_values=[0.0, 2.0, 0.0])
    assert np.allclose(scalar_law([0.5, 1.0, 1.5, 2.5]), [0.0, 2.0, 2.0, 0.0])

    alias_law = ScalarCalibrationLaw(
        "piecewise-constant",
        control_times=[0.0, 1.0, 2.0],
        control_values=[0.0, 2.0, 0.0],
    )
    assert np.allclose(alias_law([0.5, 1.5]), [0.0, 2.0])

    T0 = np.eye(4)
    T1 = _rz(np.pi / 2.0)
    T2 = _rz(np.pi)
    se3_law = SE3CalibrationLaw("piecewise_constant", control_times=[0.0, 1.0, 2.0], control_poses=np.stack([T0, T1, T2]))
    transforms = se3_law([0.5, 1.0, 1.5, 2.5])
    assert np.allclose(transforms[0], T0)
    assert np.allclose(transforms[1], T1)
    assert np.allclose(transforms[2], T1)
    assert np.allclose(transforms[3], T2)


def test_sinusoidal_law_shapes_and_start_value():
    times = np.linspace(0.0, 4.0, 9)
    scalar = ScalarCalibrationLaw("sinusoidal", t_start=0.0, t_end=4.0, value=0.1, amplitude=0.02, cycles=1.0)
    assert scalar(times).shape == times.shape
    assert np.isclose(scalar([0.0])[0], 0.1)

    se3_law = SE3CalibrationLaw("sinusoidal", t_start=0.0, t_end=4.0, reference=np.eye(4), amplitude_xi=[0.1, 0, 0, 0, 0, 0], cycles=1.0)
    transforms = se3_law(times)
    assert transforms.shape == (times.size, 4, 4)
    assert np.allclose(transforms[0], np.eye(4))
