import numpy as np

from src.calib_observability.experiments.calibration_injection import (
    CalibrationInjectionConfig,
    apply_imu_time_offset_delta,
    apply_virtual_imu_frame_rotation,
    build_injected_imu_data,
)
from src.new_college_dataset.data import IMUData
from src.transform import se3_to_se3_vectors


def _imu_stream():
    return IMUData(
        timestamps_s=np.array([1.0, 1.1, 1.2]),
        accel_mps2=np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ),
        gyro_radps=np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        ),
        name="imu0",
        frequency_hz=10.0,
    )


def test_positive_tau_injection_shifts_sensor_time_backward():
    imu = _imu_stream()
    tau_delta = 0.015

    injected = apply_imu_time_offset_delta(imu, tau_delta)

    np.testing.assert_allclose(injected.timestamps_s + tau_delta, imu.timestamps_s)
    np.testing.assert_allclose(injected.accel_mps2, imu.accel_mps2)
    np.testing.assert_allclose(injected.gyro_radps, imu.gyro_radps)


def test_virtual_imu_frame_rotation_uses_row_vector_convention():
    imu = _imu_stream()
    R_I0_I1 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    injected = apply_virtual_imu_frame_rotation(imu, R_I0_I1)

    np.testing.assert_allclose(injected.accel_mps2, imu.accel_mps2 @ R_I0_I1)
    np.testing.assert_allclose(injected.gyro_radps, imu.gyro_radps @ R_I0_I1)
    np.testing.assert_allclose(
        np.linalg.norm(injected.accel_mps2, axis=1),
        np.linalg.norm(imu.accel_mps2, axis=1),
    )


def test_combined_injection_does_not_modify_source_stream():
    imu = _imu_stream()
    config = CalibrationInjectionConfig(
        gyro_bias_delta_radps=(0.01, -0.02, 0.03),
        tau_I_delta_s=0.02,
    )

    injected = build_injected_imu_data(imu, config)

    np.testing.assert_allclose(imu.timestamps_s, np.array([1.0, 1.1, 1.2]))
    np.testing.assert_allclose(injected.timestamps_s + 0.02, imu.timestamps_s)
    np.testing.assert_allclose(
        injected.gyro_radps,
        imu.gyro_radps + np.array([0.01, -0.02, 0.03]),
    )


def test_se3_to_se3_vectors_order_is_rotation_then_translation():
    translation_only = np.eye(4)
    translation_only[:3, 3] = np.array([1.2, -0.5, 0.25])

    vector = se3_to_se3_vectors(translation_only)[0]

    np.testing.assert_allclose(vector[:3], np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(vector[3:], np.array([1.2, -0.5, 0.25]), atol=1e-12)
