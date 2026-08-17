from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path.cwd()
if REPO_ROOT.name == "tests":
    REPO_ROOT = REPO_ROOT.parent
MROB_ROOT = Path("/home/camel/Skoltech/Mobile_Robotics_Lab/mrob")
if (MROB_ROOT / "mrobpy").exists():
    sys.path.insert(0, str(MROB_ROOT / "mrobpy"))

import mrob

BUILD_COMMAND_USED = "cd build && cmake .. && make -j 24 && make -j 24 python-package"
np.random.seed(7)
np.set_printoptions(precision=6, suppress=True)

RESULTS = []
IMPLEMENTATION_BUGS_FOUND = []
BINDING_MISMATCHES_FOUND = []
IMPOSSIBLE_WITHOUT_BINDING = [
    "Exact per-factor residual/Jacobian block comparisons are not possible with the current Python API; "
    "this notebook uses graph-level chi-square and black-box perturbations instead."
]


def record(name, category, status, details="", classification=""):
    RESULTS.append(
        {
            "category": category,
            "check": name,
            "status": status,
            "classification": classification,
            "details": details,
        }
    )
    print(f"[{status}] {category}: {name}")
    if details:
        print(f"       {details}")


def check(name, category, fn, failure_classification):
    try:
        details = fn()
        record(name, category, "PASS", "" if details is None else str(details))
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        record(name, category, "FAIL", message, failure_classification)
        if failure_classification == "binding mismatch":
            BINDING_MISMATCHES_FOUND.append(f"{name}: {message}")
        if failure_classification == "implementation bug":
            IMPLEMENTATION_BUGS_FOUND.append(f"{name}: {message}")
        print(traceback.format_exc(limit=6))
        raise


def expect_raises(name, category, fn, fragment=None):
    def runner():
        try:
            fn()
        except Exception as exc:
            message = str(exc)
            if fragment is not None:
                assert fragment in message, f"expected {fragment!r}, got {message!r}"
            return f"raised {type(exc).__name__}: {message}"
        raise AssertionError("expected a Python exception, but the call completed")

    check(name, category, runner, "binding mismatch")


def state(graph, node_id):
    return np.asarray(graph.get_estimated_state()[node_id], dtype=float)


def assert_finite_states(graph):
    for node_id, value in enumerate(graph.get_estimated_state()):
        assert np.all(np.isfinite(np.asarray(value, dtype=float))), f"node {node_id} is non-finite"


def se3(rotation=(0, 0, 0), translation=(0, 0, 0)):
    xi = np.zeros(6)
    xi[:3] = np.asarray(rotation, dtype=float)
    xi[3:] = np.asarray(translation, dtype=float)
    return mrob.SE3(xi)


def se3_rt(rotation_matrix, translation=(0, 0, 0)):
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation_matrix, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return mrob.SE3(transform)


def integrate_piecewise_linear(times, values, start_time, end_time):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    assert times[0] <= start_time < end_time <= times[-1]

    def interp(query):
        upper = int(np.searchsorted(times, query, side="left"))
        if upper == 0:
            return values[0]
        if upper >= len(times):
            return values[-1]
        if times[upper] == query:
            return values[upper]
        lower = upper - 1
        alpha = (query - times[lower]) / (times[upper] - times[lower])
        return (1 - alpha) * values[lower] + alpha * values[upper]

    boundaries = [float(start_time)]
    boundaries.extend(float(t) for t in times[(times > start_time) & (times < end_time)])
    boundaries.append(float(end_time))
    integral = np.zeros(3)
    previous_time = boundaries[0]
    previous_value = interp(previous_time)
    for current_time in boundaries[1:]:
        current_value = interp(current_time)
        integral += 0.5 * (current_time - previous_time) * (previous_value + current_value)
        previous_time = current_time
        previous_value = current_value
    return integral


def angular_velocity(times):
    times = np.asarray(times, dtype=float)
    return np.vstack(
        [
            0.08 + 0.04 * np.sin(1.7 * times),
            -0.03 + 0.03 * np.cos(1.1 * times),
            0.15 + 0.02 * np.sin(2.3 * times),
        ]
    ).T


def body_pose(time):
    return se3(
        (0.08 * np.sin(0.7 * time), 0.06 * np.cos(0.9 * time), 0.25 * time + 0.05 * np.sin(1.2 * time)),
        (0.7 * time + 0.1 * np.sin(1.5 * time), 0.5 * np.sin(0.8 * time), 0.25 * np.cos(0.6 * time)),
    )


def body_rotation(time):
    return mrob.SO3(np.array([0.25 * np.sin(0.8 * time), -0.18 * np.cos(0.5 * time), 0.35 * np.sin(0.4 * time)])).R()


def low_dynamic_accepts(linear_acceleration_world, threshold=0.7):
    return float(np.linalg.norm(linear_acceleration_world)) <= threshold


def make_gyro_graph(C_vec=np.zeros(3), bias=np.zeros(3), tau=0.0, translation=np.array([0.2, -0.1, 0.3]), anchor=True):
    graph = mrob.FGraph()
    timestamps = np.linspace(-0.4, 1.4, 181)
    true_rates = angular_velocity(timestamps)
    measured_rates = true_rates + bias
    phi = integrate_piecewise_linear(timestamps, true_rates, 0.0 + tau, 1.0 + tau)
    C = mrob.SO3(C_vec)
    target_rotation = C * mrob.SO3(phi) * C.inv()
    mode = mrob.NODE_ANCHOR if anchor else mrob.NODE_STANDARD
    origin = graph.add_node_pose_3d(se3(), mrob.NODE_ANCHOR)
    target = graph.add_node_pose_3d(mrob.SE3(target_rotation, np.zeros(3)), mrob.NODE_ANCHOR)
    extrinsic = graph.add_node_pose_3d(se3(C_vec, translation), mode)
    bias_node = graph.add_node_landmark_3d(bias, mode)
    tau_node = graph.add_node_scalar(tau, mode)
    graph.add_factor_gyro_calib_prop(0.0, 1.0, timestamps, measured_rates, origin, target, extrinsic, bias_node, tau_node, np.eye(3))
    return graph, (origin, target, extrinsic, bias_node, tau_node)


def make_accel_graph(translation=np.zeros(3), normalized=False):
    graph = mrob.FGraph()
    gravity = np.array([0.0, 0.0, -9.81])
    pose_rotvec = np.array([0.2, -0.1, 0.3])
    C_vec = np.array([0.1, -0.05, 0.02])
    predicted = mrob.SO3(C_vec).R().T @ (mrob.SO3(pose_rotvec).R().T @ gravity)
    measurement = predicted / np.linalg.norm(predicted) if normalized else predicted
    timestamps = [-0.2, 0.0, 0.2]
    pose = graph.add_node_pose_3d(se3(pose_rotvec), mrob.NODE_ANCHOR)
    extrinsic = graph.add_node_pose_3d(se3(C_vec, translation), mrob.NODE_ANCHOR)
    tau = graph.add_node_scalar(0.0, mrob.NODE_ANCHOR)
    graph.add_factor_accel_gravity_calib(0.0, timestamps, np.tile(measurement, (3, 1)), gravity, pose, extrinsic, tau, np.eye(3))
    return graph, (pose, extrinsic, tau), predicted, gravity


def make_lidar_graph(tau=0.0, rotation_offset=np.zeros(3), translation_offset=np.zeros(3), anchor=True):
    graph = mrob.FGraph()
    T_B_L = se3([0.08, -0.04, 0.06], [0.3, -0.15, 0.2])
    timestamps = np.linspace(-0.3, 1.3, 81)
    lidar_poses = [body_pose(float(t - tau)) * T_B_L for t in timestamps]
    mode = mrob.NODE_ANCHOR if anchor else mrob.NODE_STANDARD
    origin = graph.add_node_pose_3d(body_pose(0.0), mrob.NODE_ANCHOR)
    target = graph.add_node_pose_3d(body_pose(1.0), mrob.NODE_ANCHOR)
    extrinsic = graph.add_node_pose_3d(se3(np.array([0.08, -0.04, 0.06]) + rotation_offset, np.array([0.3, -0.15, 0.2]) + translation_offset), mode)
    tau_node = graph.add_node_scalar(tau, mode)
    graph.add_factor_lidar_calib_odometry(0.0, 1.0, timestamps, lidar_poses, origin, target, extrinsic, tau_node, np.eye(6))
    return graph, (origin, target, extrinsic, tau_node), T_B_L


def print_graph_to_log(graph, label):
    # Exercise the C++ graph.print(True) implementation without flooding the notebook output.
    # The complete before/after graph dumps are saved next to the notebook.
    import ctypes
    import os

    log_dir = REPO_ROOT / "tests" / "calibration_pybinding_print_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{label}.txt"
    saved_stdout = os.dup(1)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            os.dup2(handle.fileno(), 1)
            graph.print(True)
            ctypes.CDLL(None).fflush(None)
    finally:
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
    return path


def run_api_checks():
    def api():
        graph = mrob.FGraph()
        methods = [
            "add_node_scalar",
            "add_factor_gyro_calib_prop",
            "add_factor_accel_gravity_calib",
            "add_factor_accel_lever_arm_calib",
            "add_factor_lidar_calib_odometry",
            "add_node_pose_3d",
            "add_node_landmark_3d",
            "get_estimated_state",
            "chi2",
            "solve",
            "print",
        ]
        symbols = ["SE3", "SO3", "NODE_STANDARD", "NODE_ANCHOR", "GN", "LM"]
        assert not [name for name in methods if not hasattr(graph, name)]
        assert not [name for name in symbols if not hasattr(mrob, name)]
        return f"mrob imported from {mrob.__file__}"

    check("requested graph methods and symbols", "API", api, "binding mismatch")
    record("analytic residual/Jacobian block access", "Binding surface", "LIMITED", IMPOSSIBLE_WITHOUT_BINDING[0], "current binding surface limitation")


def run_node_checks():
    def scalar_initial():
        graph = mrob.FGraph()
        scalar = graph.add_node_scalar(0.25)
        graph.print(True)
        assert state(graph, scalar).shape == (1, 1)
        assert np.allclose(state(graph, scalar), [[0.25]])
        assert graph.number_nodes() == 1 and graph.number_factors() == 0
        return "initial scalar state is [[0.25]]"

    def scalar_updates_indirectly():
        base, _, predicted, gravity = make_accel_graph()
        slope = np.array([1.0, -0.5, 0.2])
        tau_truth = 0.12
        timestamps = np.linspace(-1.0, 1.0, 81)
        measurements = np.vstack([predicted + slope * (t - tau_truth) for t in timestamps])
        graph = mrob.FGraph()
        pose = graph.add_node_pose_3d(se3([0.2, -0.1, 0.3]), mrob.NODE_ANCHOR)
        extrinsic = graph.add_node_pose_3d(se3([0.1, -0.05, 0.02]), mrob.NODE_ANCHOR)
        tau = graph.add_node_scalar(0.0, mrob.NODE_STANDARD)
        graph.add_factor_accel_gravity_calib(0.0, timestamps, measurements, gravity, pose, extrinsic, tau, 10 * np.eye(3))
        before = float(graph.chi2())
        graph.solve(mrob.LM, 30)
        after = float(graph.chi2())
        tau_after = float(state(graph, tau)[0, 0])
        assert abs(tau_after - tau_truth) < 1e-7
        assert after < before * 1e-8
        return f"tau 0 -> {tau_after:.6f}, chi2 {before:.3e} -> {after:.3e}"

    def anchored_scalar():
        _, _, predicted, gravity = make_accel_graph()
        slope = np.array([1.0, -0.5, 0.2])
        timestamps = np.linspace(-1.0, 1.0, 81)
        measurements = np.vstack([predicted + slope * (t - 0.12) for t in timestamps])
        graph = mrob.FGraph()
        pose = graph.add_node_pose_3d(se3([0.2, -0.1, 0.3]), mrob.NODE_ANCHOR)
        extrinsic = graph.add_node_pose_3d(se3([0.1, -0.05, 0.02]), mrob.NODE_ANCHOR)
        tau = graph.add_node_scalar(0.0, mrob.NODE_ANCHOR)
        graph.add_factor_accel_gravity_calib(0.0, timestamps, measurements, gravity, pose, extrinsic, tau, 10 * np.eye(3))
        before_tau = float(state(graph, tau)[0, 0])
        before = float(graph.chi2())
        graph.solve(mrob.LM, 10)
        assert float(state(graph, tau)[0, 0]) == before_tau
        assert abs(float(graph.chi2()) - before) < 1e-12
        return "anchored tau remained unchanged"

    def pose3d_dual_use():
        graph, nodes = make_gyro_graph()
        assert state(graph, nodes[0]).shape == (4, 4)
        assert state(graph, nodes[2]).shape == (4, 4)
        assert graph.number_nodes() == 5 and graph.number_factors() == 1
        return "NodePose3d works for trajectory poses and T_B_I"

    check("initial value, shape, and print(True)", "NodeScalar", scalar_initial, "binding mismatch")
    check("additive update through calibration factor", "NodeScalar", scalar_updates_indirectly, "implementation bug")
    check("anchored scalar remains unchanged", "NodeScalar", anchored_scalar, "implementation bug")
    check("NodePose3d as trajectory and extrinsic", "NodePose3d", pose3d_dual_use, "binding mismatch")


def run_smoke_tests():
    def gyro():
        graph, _ = make_gyro_graph()
        chi = float(graph.chi2())
        graph.print(True)
        graph.solve(mrob.GN, 1)
        assert chi < 1e-18
        assert graph.number_nodes() == 5 and graph.number_factors() == 1
        assert_finite_states(graph)
        return f"chi2={chi:.3e}"

    def accel():
        graph, _, _, _ = make_accel_graph()
        chi = float(graph.chi2())
        graph.print(True)
        graph.solve(mrob.GN, 1)
        assert chi < 1e-24
        assert graph.number_nodes() == 3 and graph.number_factors() == 1
        assert_finite_states(graph)
        return f"chi2={chi:.3e}"

    def lidar():
        graph, _, _ = make_lidar_graph()
        chi = float(graph.chi2())
        graph.print(True)
        graph.solve(mrob.GN, 1)
        assert chi < 1e-24
        assert graph.number_nodes() == 4 and graph.number_factors() == 1
        assert_finite_states(graph)
        return f"chi2={chi:.3e}"

    check("FactorGyroCalibProp minimal graph", "Smoke", gyro, "implementation bug")
    check("FactorAccelGravityCalib minimal graph", "Smoke", accel, "implementation bug")
    check("FactorLidarCalibOdometry minimal graph", "Smoke", lidar, "implementation bug")


def validation_nodes(kind):
    graph = mrob.FGraph()
    if kind == "gyro":
        nodes = (
            graph.add_node_pose_3d(se3(), mrob.NODE_ANCHOR),
            graph.add_node_pose_3d(se3([0.1, 0, 0]), mrob.NODE_ANCHOR),
            graph.add_node_pose_3d(se3(), mrob.NODE_ANCHOR),
            graph.add_node_landmark_3d(np.zeros(3), mrob.NODE_ANCHOR),
            graph.add_node_scalar(0.0, mrob.NODE_ANCHOR),
        )
    elif kind == "accel":
        nodes = (
            graph.add_node_pose_3d(se3(), mrob.NODE_ANCHOR),
            graph.add_node_pose_3d(se3(), mrob.NODE_ANCHOR),
            graph.add_node_scalar(0.0, mrob.NODE_ANCHOR),
        )
    else:
        nodes = (
            graph.add_node_pose_3d(se3(), mrob.NODE_ANCHOR),
            graph.add_node_pose_3d(se3([0.1, 0, 0], [1, 0, 0]), mrob.NODE_ANCHOR),
            graph.add_node_pose_3d(se3(), mrob.NODE_ANCHOR),
            graph.add_node_scalar(0.0, mrob.NODE_ANCHOR),
        )
    return graph, nodes


def run_input_validation_tests():
    def add_gyro(timestamps=None, measurements=None):
        graph, nodes = validation_nodes("gyro")
        timestamps = [0.0, 0.5, 1.0] if timestamps is None else timestamps
        measurements = np.zeros((len(timestamps), 3)) if measurements is None else measurements
        graph.add_factor_gyro_calib_prop(0.0, 1.0, timestamps, measurements, *nodes, np.eye(3))
        return graph

    def add_accel(timestamps=None, measurements=None):
        graph, nodes = validation_nodes("accel")
        timestamps = [-0.5, 0.0, 0.5] if timestamps is None else timestamps
        measurements = np.tile([0.0, 0.0, 9.81], (len(timestamps), 1)) if measurements is None else measurements
        graph.add_factor_accel_gravity_calib(0.0, timestamps, measurements, np.array([0.0, 0.0, -9.81]), *nodes, np.eye(3))
        return graph

    def add_lidar(timestamps=None, poses=None):
        graph, nodes = validation_nodes("lidar")
        timestamps = [0.0, 0.5, 1.0] if timestamps is None else timestamps
        poses = [se3([0, 0, 0.1 * t], [t, 0, 0]) for t in timestamps] if poses is None else poses
        graph.add_factor_lidar_calib_odometry(0.0, 1.0, timestamps, poses, *nodes, np.eye(6))
        return graph

    expect_raises("gyro array not shaped (N,3)", "Input validation", lambda: add_gyro(measurements=np.zeros((3, 2))), "shape (N, 3)")
    expect_raises("accelerometer array not shaped (N,3)", "Input validation", lambda: add_accel(measurements=np.zeros((3, 2))), "shape (N, 3)")
    expect_raises("gyro timestamp/measurement length mismatch", "Input validation", lambda: add_gyro([0.0, 0.5], np.zeros((3, 3))), "same number of rows")
    expect_raises("accelerometer timestamp/measurement length mismatch", "Input validation", lambda: add_accel([0.0, 0.5], np.zeros((3, 3))), "same number of rows")
    expect_raises("LiDAR timestamp/pose length mismatch", "Input validation", lambda: add_lidar([0.0, 0.5], [mrob.SE3(), mrob.SE3(), mrob.SE3()]), "same length")
    expect_raises("gyro fewer than two samples", "Input validation", lambda: add_gyro([0.0], np.zeros((1, 3))), "at least two")
    expect_raises("accelerometer fewer than two samples", "Input validation", lambda: add_accel([0.0], np.zeros((1, 3))), "at least two")
    expect_raises("LiDAR fewer than two poses", "Input validation", lambda: add_lidar([0.0], [mrob.SE3()]), "at least two")
    expect_raises("gyro non-increasing timestamps", "Input validation", lambda: add_gyro([0.0, 0.5, 0.5], np.zeros((3, 3))), "strictly increasing")
    expect_raises("accelerometer non-increasing timestamps", "Input validation", lambda: add_accel([0.0, 0.0, 0.5], np.zeros((3, 3))), "strictly increasing")
    expect_raises("LiDAR non-increasing timestamps", "Input validation", lambda: add_lidar([0.0, 0.5, 0.5], [mrob.SE3(), mrob.SE3(), mrob.SE3()]), "strictly increasing")
    invalid_pose = np.eye(4)
    invalid_pose[0, 0] = 2.0
    expect_raises("invalid LiDAR pose matrix", "Input validation", lambda: add_lidar([0.0, 1.0], [np.eye(4), invalid_pose]), "valid SE(3)")

    def gyro_out():
        graph = add_gyro()
        graph.get_estimated_state()[4][0, 0] = 0.4

    def gyro_out_real():
        graph = mrob.FGraph()
        origin = graph.add_node_pose_3d(se3(), mrob.NODE_ANCHOR)
        target = graph.add_node_pose_3d(se3([0.1, 0, 0]), mrob.NODE_ANCHOR)
        extrinsic = graph.add_node_pose_3d(se3(), mrob.NODE_ANCHOR)
        bias = graph.add_node_landmark_3d(np.zeros(3), mrob.NODE_ANCHOR)
        tau = graph.add_node_scalar(0.4, mrob.NODE_ANCHOR)
        graph.add_factor_gyro_calib_prop(0.0, 1.0, [0.0, 0.5, 1.0], np.zeros((3, 3)), origin, target, extrinsic, bias, tau, np.eye(3))
        graph.chi2()

    def accel_out():
        graph = mrob.FGraph()
        pose = graph.add_node_pose_3d(se3(), mrob.NODE_ANCHOR)
        extrinsic = graph.add_node_pose_3d(se3(), mrob.NODE_ANCHOR)
        tau = graph.add_node_scalar(1.0, mrob.NODE_ANCHOR)
        graph.add_factor_accel_gravity_calib(0.0, [-0.2, 0.0, 0.2], np.tile([0, 0, 9.81], (3, 1)), np.array([0, 0, -9.81]), pose, extrinsic, tau, np.eye(3))
        graph.chi2()

    def lidar_out():
        graph = mrob.FGraph()
        origin = graph.add_node_pose_3d(se3(), mrob.NODE_ANCHOR)
        target = graph.add_node_pose_3d(se3([0, 0, 0.1], [1, 0, 0]), mrob.NODE_ANCHOR)
        extrinsic = graph.add_node_pose_3d(se3(), mrob.NODE_ANCHOR)
        tau = graph.add_node_scalar(0.4, mrob.NODE_ANCHOR)
        poses = [mrob.SE3(), se3([0, 0, 0.05], [0.5, 0, 0]), se3([0, 0, 0.1], [1, 0, 0])]
        graph.add_factor_lidar_calib_odometry(0.0, 1.0, [0.0, 0.5, 1.0], poses, origin, target, extrinsic, tau, np.eye(6))
        graph.chi2()

    expect_raises("gyro shifted interval outside support", "Input validation", gyro_out_real, "outside gyroscope support")
    expect_raises("accelerometer shifted query outside support", "Input validation", accel_out, "outside sample support")
    expect_raises("LiDAR shifted query outside support", "Input validation", lidar_out, "outside measurement support")


def run_regression_tests():
    def gyro_constant():
        w = np.array([0.1, -0.03, 0.2])
        timestamps = np.linspace(-0.2, 1.2, 15)
        measurements = np.tile(w, (len(timestamps), 1))
        new = mrob.FGraph()
        n0 = new.add_node_pose_3d(se3(), mrob.NODE_ANCHOR)
        n1 = new.add_node_pose_3d(mrob.SE3(mrob.SO3(w), np.zeros(3)), mrob.NODE_ANCHOR)
        nC = new.add_node_pose_3d(se3(), mrob.NODE_ANCHOR)
        nb = new.add_node_landmark_3d(np.zeros(3), mrob.NODE_ANCHOR)
        nt = new.add_node_scalar(0.0, mrob.NODE_ANCHOR)
        new.add_factor_gyro_calib_prop(0.0, 1.0, timestamps, measurements, n0, n1, nC, nb, nt, np.eye(3))
        old = mrob.FGraph()
        o0 = old.add_node_so3(mrob.SO3(), mrob.NODE_ANCHOR)
        o1 = old.add_node_so3(mrob.SO3(w), mrob.NODE_ANCHOR)
        old.add_factor_gyro_prop(w, 1.0, o0, o1, np.eye(3))
        assert float(new.chi2()) < 1e-24 and float(old.chi2()) < 1e-24
        assert abs(float(new.chi2()) - float(old.chi2())) < 1e-24
        return f"new={float(new.chi2()):.3e}, old={float(old.chi2()):.3e}"

    def gyro_bias():
        w = np.array([0.1, -0.03, 0.2])
        b = np.array([0.01, -0.02, 0.03])
        timestamps = np.linspace(-0.2, 1.2, 15)
        measurements = np.tile(w + b, (len(timestamps), 1))
        new = mrob.FGraph()
        n0 = new.add_node_pose_3d(se3(), mrob.NODE_ANCHOR)
        n1 = new.add_node_pose_3d(mrob.SE3(mrob.SO3(w), np.zeros(3)), mrob.NODE_ANCHOR)
        nC = new.add_node_pose_3d(se3(), mrob.NODE_ANCHOR)
        nb = new.add_node_landmark_3d(b, mrob.NODE_ANCHOR)
        nt = new.add_node_scalar(0.0, mrob.NODE_ANCHOR)
        new.add_factor_gyro_calib_prop(0.0, 1.0, timestamps, measurements, n0, n1, nC, nb, nt, np.eye(3))
        old = mrob.FGraph()
        o0 = old.add_node_so3(mrob.SO3(), mrob.NODE_ANCHOR)
        o1 = old.add_node_so3(mrob.SO3(w), mrob.NODE_ANCHOR)
        ob = old.add_node_landmark_3d(b, mrob.NODE_ANCHOR)
        old.add_factor_gyro_bias_prop(w + b, 1.0, o0, o1, ob, np.eye(3))
        assert float(new.chi2()) < 1e-24 and float(old.chi2()) < 1e-24
        assert abs(float(new.chi2()) - float(old.chi2())) < 1e-24
        return f"new={float(new.chi2()):.3e}, old={float(old.chi2()):.3e}"

    def gyro_rotated():
        w = np.array([0.1, -0.03, 0.2])
        b = np.array([0.01, -0.02, 0.03])
        C_vec = np.array([0.2, 0.1, -0.1])
        C = mrob.SO3(C_vec)
        target = C * mrob.SO3(w) * C.inv()
        timestamps = np.linspace(-0.2, 1.2, 15)
        measurements = np.tile(w + b, (len(timestamps), 1))
        new = mrob.FGraph()
        n0 = new.add_node_pose_3d(se3(), mrob.NODE_ANCHOR)
        n1 = new.add_node_pose_3d(mrob.SE3(target, np.zeros(3)), mrob.NODE_ANCHOR)
        nC = new.add_node_pose_3d(se3(C_vec, [0.4, -0.2, 0.1]), mrob.NODE_ANCHOR)
        nb = new.add_node_landmark_3d(b, mrob.NODE_ANCHOR)
        nt = new.add_node_scalar(0.0, mrob.NODE_ANCHOR)
        new.add_factor_gyro_calib_prop(0.0, 1.0, timestamps, measurements, n0, n1, nC, nb, nt, np.eye(3))
        old = mrob.FGraph()
        o0 = old.add_node_so3(mrob.SO3(), mrob.NODE_ANCHOR)
        o1 = old.add_node_so3(target, mrob.NODE_ANCHOR)
        ob = old.add_node_landmark_3d(b, mrob.NODE_ANCHOR)
        oC = old.add_node_so3(C, mrob.NODE_ANCHOR)
        old.add_factor_rotated_gyro_bias_prop(w + b, 1.0, o0, o1, ob, oC, np.eye(3))
        assert float(new.chi2()) < 1e-24 and float(old.chi2()) < 1e-24
        assert abs(float(new.chi2()) - float(old.chi2())) < 1e-24
        return f"new={float(new.chi2()):.3e}, old={float(old.chi2()):.3e}"

    def gyro_translation_invariant():
        a, _ = make_gyro_graph(np.array([0.15, -0.04, 0.07]), translation=np.zeros(3))
        b, _ = make_gyro_graph(np.array([0.15, -0.04, 0.07]), translation=np.array([1.5, -2.0, 0.8]))
        assert abs(float(a.chi2()) - float(b.chi2())) < 1e-20
        return f"{float(a.chi2()):.3e} vs {float(b.chi2()):.3e}"

    def accel_magnitude():
        graph, _, predicted, _ = make_accel_graph(normalized=True)
        chi = float(graph.chi2())
        assert np.isclose(np.linalg.norm(predicted), 9.81)
        assert chi > 30.0
        return f"|truth|={np.linalg.norm(predicted):.3f}, normalized-input chi2={chi:.3e}"

    def accel_translation():
        a, _, _, _ = make_accel_graph(np.zeros(3))
        b, _, _, _ = make_accel_graph(np.array([2.0, -1.0, 0.5]))
        assert float(a.chi2()) < 1e-24
        assert abs(float(a.chi2()) - float(b.chi2())) < 1e-20
        return f"{float(a.chi2()):.3e} vs {float(b.chi2()):.3e}"

    def accel_gating():
        samples = [np.array([0.05, 0.02, 0.0]), np.array([3.0, 0.0, 0.0]), np.array([0.1, -0.1, 0.1])]
        mask = [low_dynamic_accepts(sample) for sample in samples]
        assert mask == [True, False, True]
        return f"accepted mask={mask}"

    def accel_tau_lookup():
        _, _, predicted, gravity = make_accel_graph()
        slope = np.array([1.0, -0.5, 0.2])
        tau_truth = 0.12
        timestamps = np.linspace(-1.0, 1.0, 81)
        measurements = np.vstack([predicted + slope * (t - tau_truth) for t in timestamps])

        def chi(tau_value):
            graph = mrob.FGraph()
            pose = graph.add_node_pose_3d(se3([0.2, -0.1, 0.3]), mrob.NODE_ANCHOR)
            extrinsic = graph.add_node_pose_3d(se3([0.1, -0.05, 0.02]), mrob.NODE_ANCHOR)
            tau = graph.add_node_scalar(tau_value, mrob.NODE_ANCHOR)
            graph.add_factor_accel_gravity_calib(0.0, timestamps, measurements, gravity, pose, extrinsic, tau, np.eye(3))
            pose_before = state(graph, pose).copy()
            out = float(graph.chi2())
            assert np.allclose(pose_before, state(graph, pose))
            return out

        truth = chi(tau_truth)
        wrong = chi(0.0)
        assert truth < 1e-24 and wrong > truth + 1e-3
        return f"truth={truth:.3e}, wrong={wrong:.3e}"

    def lidar_sensitivity():
        truth, _, _ = make_lidar_graph()
        rot, _, _ = make_lidar_graph(rotation_offset=np.array([0.05, 0, 0]))
        trans, _, _ = make_lidar_graph(translation_offset=np.array([0.1, 0, 0]))
        assert float(truth.chi2()) < 1e-24
        assert float(rot.chi2()) > float(truth.chi2()) + 1e-5
        assert float(trans.chi2()) > float(truth.chi2()) + 1e-5
        return f"truth={float(truth.chi2()):.3e}, rot={float(rot.chi2()):.3e}, trans={float(trans.chi2()):.3e}"

    def lidar_tau_lookup():
        tau_truth = 0.2
        T_B_L = se3([0.08, -0.04, 0.06], [0.3, -0.15, 0.2])
        timestamps = np.linspace(-0.3, 1.5, 181)
        lidar_poses = [body_pose(float(t - tau_truth)) * T_B_L for t in timestamps]

        def build_with_node_tau(node_tau_value):
            graph = mrob.FGraph()
            origin = graph.add_node_pose_3d(body_pose(0.0), mrob.NODE_ANCHOR)
            target = graph.add_node_pose_3d(body_pose(1.0), mrob.NODE_ANCHOR)
            extrinsic = graph.add_node_pose_3d(T_B_L, mrob.NODE_ANCHOR)
            tau_node = graph.add_node_scalar(node_tau_value, mrob.NODE_ANCHOR)
            graph.add_factor_lidar_calib_odometry(0.0, 1.0, timestamps, lidar_poses, origin, target, extrinsic, tau_node, np.eye(6))
            pose_before = state(graph, origin).copy()
            chi = float(graph.chi2())
            assert np.allclose(pose_before, state(graph, origin))
            return chi

        truth_chi = build_with_node_tau(tau_truth)
        wrong_chi = build_with_node_tau(0.0)
        assert truth_chi < 1e-20 and wrong_chi > truth_chi + 1e-5
        return f"truth={truth_chi:.3e}, wrong={wrong_chi:.3e}"

    check("gyro calibrated factor matches FactorGyroProp", "Regression gyro", gyro_constant, "sign or frame-convention mismatch")
    check("gyro calibrated factor matches FactorGyroBiasProp", "Regression gyro", gyro_bias, "sign or frame-convention mismatch")
    check("gyro calibrated factor matches FactorRotatedGyroBiasProp", "Regression gyro", gyro_rotated, "sign or frame-convention mismatch")
    check("gyro chi-square invariant to T_B_I translation", "Regression gyro", gyro_translation_invariant, "implementation bug")
    check("accelerometer magnitude is preserved", "Regression accel", accel_magnitude, "implementation bug")
    check("accelerometer truth and T_B_I translation invariance", "Regression accel", accel_translation, "implementation bug")
    check("Python-side low-dynamic gating rejects dynamic samples", "Regression accel", accel_gating, "insufficient excitation or sample rejected by design")
    check("accelerometer tau shifts measurement lookup only", "Regression accel", accel_tau_lookup, "implementation bug")
    check("LiDAR truth and extrinsic sensitivity", "Regression lidar", lidar_sensitivity, "sign or frame-convention mismatch")
    check("LiDAR tau shifts measurement lookup only", "Regression lidar", lidar_tau_lookup, "implementation bug")


def run_convergence_tests():
    def gyro_conv():
        graph = mrob.FGraph()
        C_truth = np.array([0.12, -0.08, 0.05])
        trans_truth = np.array([0.1, 0.2, -0.05])
        bias_truth = np.array([0.015, -0.02, 0.01])
        tau_truth = 0.03
        timestamps = np.linspace(-0.2, 4.4, 461)
        rates = angular_velocity(timestamps)
        measurements = rates + bias_truth
        pose_times = np.linspace(0.0, 4.0, 9)
        pose_ids = []
        R = mrob.SO3()
        C = mrob.SO3(C_truth)
        pose_ids.append(graph.add_node_pose_3d(mrob.SE3(R, np.zeros(3)), mrob.NODE_ANCHOR))
        for a, b in zip(pose_times[:-1], pose_times[1:]):
            phi = integrate_piecewise_linear(timestamps, rates, a + tau_truth, b + tau_truth)
            R = R * (C * mrob.SO3(phi) * C.inv())
            pose_ids.append(graph.add_node_pose_3d(mrob.SE3(R, np.zeros(3)), mrob.NODE_ANCHOR))
        nC = graph.add_node_pose_3d(se3(C_truth + [0.03, -0.02, 0.02], trans_truth), mrob.NODE_STANDARD)
        nb = graph.add_node_landmark_3d(bias_truth + [0.01, -0.005, 0.006], mrob.NODE_STANDARD)
        nt = graph.add_node_scalar(tau_truth + 0.015, mrob.NODE_STANDARD)
        anchored_before = [state(graph, node).copy() for node in pose_ids]
        before_rot = mrob.SE3(state(graph, nC)).Ln()[:3]
        before_bias = state(graph, nb).ravel()
        before_tau = float(state(graph, nt)[0, 0])
        for o, t, a, b in zip(pose_ids[:-1], pose_ids[1:], pose_times[:-1], pose_times[1:]):
            graph.add_factor_gyro_calib_prop(float(a), float(b), timestamps, measurements, o, t, nC, nb, nt, 10 * np.eye(3))
        before = float(graph.chi2())
        before_log = print_graph_to_log(graph, "gyro_only_before")
        graph.solve(mrob.LM, 80, 1e-4, 1e-9, False)
        after_log = print_graph_to_log(graph, "gyro_only_after")
        after = float(graph.chi2())
        after_rot = mrob.SE3(state(graph, nC)).Ln()[:3]
        after_bias = state(graph, nb).ravel()
        after_tau = float(state(graph, nt)[0, 0])
        for old, node in zip(anchored_before, pose_ids):
            assert np.allclose(old, state(graph, node), atol=1e-12)
        assert before > after * 1e6
        assert np.linalg.norm(after_rot - before_rot) > 1e-3
        assert np.linalg.norm(after_bias - before_bias) > 1e-3
        assert abs(after_tau - before_tau) > 1e-3
        assert np.linalg.norm(after_rot - C_truth) < 5e-5
        assert np.linalg.norm(after_bias - bias_truth) < 5e-5
        assert abs(after_tau - tau_truth) < 5e-5
        return f"chi2 {before:.3e}->{after:.3e}; rot_err={np.linalg.norm(after_rot-C_truth):.3e}, bias_err={np.linalg.norm(after_bias-bias_truth):.3e}, tau_err={abs(after_tau-tau_truth):.3e}"

    def accel_conv():
        graph = mrob.FGraph()
        gravity = np.array([0.0, 0.0, -9.81])
        C_truth = np.array([0.12, -0.07, 0.05])
        tau_truth = 0.04
        timestamps = np.linspace(-0.2, 5.2, 541)
        measurements = np.vstack([mrob.SO3(C_truth).R().T @ (body_rotation(t - tau_truth).T @ gravity) for t in timestamps])
        pose_times = np.linspace(0.1, 5.0, 25)
        pose_ids = [graph.add_node_pose_3d(mrob.SE3(mrob.SO3(body_rotation(t)), np.zeros(3)), mrob.NODE_ANCHOR) for t in pose_times]
        nC = graph.add_node_pose_3d(se3(C_truth + [0.05, -0.04, 0.03], [0.2, -0.1, 0.3]), mrob.NODE_STANDARD)
        nt = graph.add_node_scalar(tau_truth + 0.02, mrob.NODE_STANDARD)
        anchored_before = [state(graph, node).copy() for node in pose_ids]
        before_rot = mrob.SE3(state(graph, nC)).Ln()[:3]
        before_tau = float(state(graph, nt)[0, 0])
        for t, pose in zip(pose_times, pose_ids):
            graph.add_factor_accel_gravity_calib(float(t), timestamps, measurements, gravity, pose, nC, nt, 0.1 * np.eye(3))
        before = float(graph.chi2())
        before_log = print_graph_to_log(graph, "accel_only_before")
        graph.solve(mrob.LM, 100, 1e-4, 1e-9, False)
        after_log = print_graph_to_log(graph, "accel_only_after")
        after = float(graph.chi2())
        after_rot = mrob.SE3(state(graph, nC)).Ln()[:3]
        after_tau = float(state(graph, nt)[0, 0])
        for old, node in zip(anchored_before, pose_ids):
            assert np.allclose(old, state(graph, node), atol=1e-12)
        assert before > after * 1e6
        assert np.linalg.norm(after_rot - before_rot) > 1e-3
        assert abs(after_tau - before_tau) > 1e-3
        assert np.linalg.norm(after_rot - C_truth) < 1e-4
        assert abs(after_tau - tau_truth) < 5e-4
        return f"chi2 {before:.3e}->{after:.3e}; rot_err={np.linalg.norm(after_rot-C_truth):.3e}, tau_err={abs(after_tau-tau_truth):.3e}; translation unobservable"

    def lidar_conv():
        graph = mrob.FGraph()
        truth = se3([0.08, -0.04, 0.06], [0.3, -0.15, 0.2])
        tau_truth = 0.05
        timestamps = np.linspace(-0.2, 6.2, 641)
        lidar_poses = [body_pose(float(t - tau_truth)) * truth for t in timestamps]
        pose_times = np.linspace(0.2, 5.8, 15)
        pose_ids = [graph.add_node_pose_3d(body_pose(float(t)), mrob.NODE_ANCHOR) for t in pose_times]
        nL = graph.add_node_pose_3d(se3([0.11, -0.06, 0.08], [0.35, -0.12, 0.16]), mrob.NODE_STANDARD)
        nt = graph.add_node_scalar(0.02, mrob.NODE_STANDARD)
        anchored_before = [state(graph, node).copy() for node in pose_ids]
        before_ext = mrob.SE3(state(graph, nL))
        before_tau = float(state(graph, nt)[0, 0])
        for o, t, a, b in zip(pose_ids[:-1], pose_ids[1:], pose_times[:-1], pose_times[1:]):
            graph.add_factor_lidar_calib_odometry(float(a), float(b), timestamps, lidar_poses, o, t, nL, nt, np.eye(6))
        before = float(graph.chi2())
        before_log = print_graph_to_log(graph, "lidar_only_before")
        graph.solve(mrob.LM, 100, 1e-4, 1e-9, False)
        after_log = print_graph_to_log(graph, "lidar_only_after")
        after = float(graph.chi2())
        after_ext = mrob.SE3(state(graph, nL))
        after_tau = float(state(graph, nt)[0, 0])
        for old, node in zip(anchored_before, pose_ids):
            assert np.allclose(old, state(graph, node), atol=1e-12)
        assert before > after * 1e6
        assert after_ext.distance(before_ext) > 1e-3
        assert abs(after_tau - before_tau) > 1e-3
        assert after_ext.distance_rotation(truth) < 1e-5
        assert after_ext.distance_trans(truth) < 1e-5
        assert abs(after_tau - tau_truth) < 1e-5
        return f"chi2 {before:.3e}->{after:.3e}; rot_err={after_ext.distance_rotation(truth):.3e}, trans_err={after_ext.distance_trans(truth):.3e}, tau_err={abs(after_tau-tau_truth):.3e}"

    check("gyro-only graph converges in observable variables", "Isolated convergence", gyro_conv, "insufficient excitation or unobservable variable")
    check("accelerometer-only graph converges in observable variables", "Isolated convergence", accel_conv, "insufficient excitation or unobservable variable")
    check("LiDAR-only graph converges in full T_B_L and tau_L", "Isolated convergence", lidar_conv, "insufficient excitation or unobservable variable")



def _quadratic_position(time, acceleration=np.array([0.7, -0.35, 0.22])):
    time = float(time)
    q0 = np.array([0.25, -0.4, 0.15])
    v0 = np.array([0.45, 0.2, -0.08])
    return q0 + v0 * time + 0.5 * np.asarray(acceleration, dtype=float) * time * time


def _accel_lever_case(
    lever_arm=np.array([0.45, -0.2, 0.18]),
    omega_at_query=np.array([0.35, -0.2, 0.28]),
    alpha_body=np.array([0.7, -0.35, 0.25]),
    force_slope=np.array([0.6, -0.4, 0.25]),
    tau_truth=0.037,
    acceleration_world=np.array([0.7, -0.35, 0.22]),
):
    gravity = np.array([0.0, 0.0, -9.81])
    pose_times = np.array([-0.18, 0.07, 0.36])
    pose_time = float(pose_times[1])
    query_truth = pose_time + float(tau_truth)
    C_vec = np.array([0.16, -0.11, 0.08])
    C = mrob.SO3(C_vec).R()
    bias = np.array([0.015, -0.02, 0.012])
    acceleration_world = np.asarray(acceleration_world, dtype=float)
    positions = [_quadratic_position(t, acceleration_world) for t in pose_times]
    R_pose = body_rotation(pose_time)
    T_B_I = se3_rt(C, lever_arm)

    def omega_body(sensor_time):
        return np.asarray(omega_at_query, dtype=float) + np.asarray(alpha_body, dtype=float) * (float(sensor_time) - query_truth)

    timestamps = np.linspace(pose_time - 0.45, pose_time + 0.45, 91)
    gyroscope = np.vstack([C.T @ omega_body(t) + bias for t in timestamps])
    omega_q = omega_body(query_truth)
    alpha_q = np.asarray(alpha_body, dtype=float)
    lever_accel = np.cross(alpha_q, lever_arm) + np.cross(omega_q, np.cross(omega_q, lever_arm))
    body_origin_specific_force = R_pose.T @ (gravity - acceleration_world)
    predicted = C.T @ (body_origin_specific_force - lever_accel)
    accelerometer = np.vstack([predicted + np.asarray(force_slope, dtype=float) * (t - query_truth) for t in timestamps])

    poses = [se3_rt(R_pose if i == 1 else body_rotation(float(t)), positions[i]) for i, t in enumerate(pose_times)]
    return {
        "pose_times": pose_times,
        "pose_time": pose_time,
        "timestamps": timestamps,
        "accelerometer": accelerometer,
        "gyroscope": gyroscope,
        "gravity": gravity,
        "poses": poses,
        "T_B_I": T_B_I,
        "bias": bias,
        "tau_truth": float(tau_truth),
        "predicted": predicted,
        "lever_arm": np.asarray(lever_arm, dtype=float),
        "omega_at_query": np.asarray(omega_at_query, dtype=float),
        "alpha_body": np.asarray(alpha_body, dtype=float),
        "force_slope": np.asarray(force_slope, dtype=float),
    }


def make_accel_lever_arm_graph(case=None, tau_initial=None, lever_arm_override=None, bias_override=None, anchor_calibration=True):
    case = _accel_lever_case() if case is None else case
    graph = mrob.FGraph()
    mode = mrob.NODE_ANCHOR if anchor_calibration else mrob.NODE_STANDARD
    pose_nodes = [graph.add_node_pose_3d(pose, mrob.NODE_ANCHOR) for pose in case["poses"]]
    T_B_I = case["T_B_I"] if lever_arm_override is None else se3_rt(case["T_B_I"].R(), lever_arm_override)
    nC = graph.add_node_pose_3d(T_B_I, mode)
    nb = graph.add_node_landmark_3d(case["bias"] if bias_override is None else bias_override, mode)
    nt = graph.add_node_scalar(case["tau_truth"] if tau_initial is None else tau_initial, mode)
    factor_id = graph.add_factor_accel_lever_arm_calib(
        float(case["pose_times"][0]),
        float(case["pose_times"][1]),
        float(case["pose_times"][2]),
        case["timestamps"],
        case["accelerometer"],
        case["gyroscope"],
        case["gravity"],
        pose_nodes[0],
        pose_nodes[1],
        pose_nodes[2],
        nC,
        nb,
        nt,
        np.eye(3),
    )
    return graph, (*pose_nodes, nC, nb, nt), factor_id, case


def run_accel_lever_arm_appendix_tests():
    """Run Python-level appendix checks for FactorAccelLeverArmCalib."""

    def api_available():
        graph = mrob.FGraph()
        assert hasattr(graph, "add_factor_accel_lever_arm_calib")
        doc = graph.add_factor_accel_lever_arm_calib.__doc__ or ""
        assert "poseTimePrevious" in doc and "angularVelocityImu" in doc
        return "add_factor_accel_lever_arm_calib is exposed"

    def smoke_graph():
        graph, nodes, _, _ = make_accel_lever_arm_graph()
        chi_before = float(graph.chi2())
        log_path = print_graph_to_log(graph, "accel_lever_arm_smoke")
        graph.solve(mrob.GN, 1)
        chi_after = float(graph.chi2())
        assert chi_before < 1e-18 and chi_after < 1e-18
        assert graph.number_nodes() == 6 and graph.number_factors() == 1
        assert_finite_states(graph)
        return f"chi2={chi_before:.3e}->{chi_after:.3e}; factor log={log_path}"

    def validation_nodes_lever(tau_value=0.0):
        graph = mrob.FGraph()
        nodes = [graph.add_node_pose_3d(se3(translation=[i, 0, 0]), mrob.NODE_ANCHOR) for i in range(3)]
        nC = graph.add_node_pose_3d(se3(translation=[0.2, 0.0, 0.0]), mrob.NODE_ANCHOR)
        nb = graph.add_node_landmark_3d(np.zeros(3), mrob.NODE_ANCHOR)
        nt = graph.add_node_scalar(float(tau_value), mrob.NODE_ANCHOR)
        return graph, (*nodes, nC, nb, nt)

    def add_invalid(pose_times=(-0.1, 0.0, 0.1), timestamps=None, acc=None, gyr=None, tau=0.0):
        graph, nodes = validation_nodes_lever(tau)
        timestamps = np.array([-0.2, 0.0, 0.2]) if timestamps is None else np.asarray(timestamps, dtype=float)
        acc = np.zeros((len(timestamps), 3)) if acc is None else acc
        gyr = np.zeros((len(timestamps), 3)) if gyr is None else gyr
        graph.add_factor_accel_lever_arm_calib(*map(float, pose_times), timestamps, acc, gyr, np.array([0.0, 0.0, -9.81]), *nodes, np.eye(3))
        return graph

    check("FactorAccelLeverArmCalib API availability", "Accel lever-arm", api_available, "binding mismatch")
    check("FactorAccelLeverArmCalib minimal truth graph", "Accel lever-arm", smoke_graph, "implementation bug")

    expect_raises("lever-arm non-increasing pose times", "Accel lever-arm validation", lambda: add_invalid(pose_times=(0.0, 0.0, 0.1)), "strictly increasing")
    expect_raises("lever-arm mismatched accelerometer length", "Accel lever-arm validation", lambda: add_invalid(acc=np.zeros((4, 3))), "same number")
    expect_raises("lever-arm mismatched gyroscope length", "Accel lever-arm validation", lambda: add_invalid(gyr=np.zeros((4, 3))), "same number")
    expect_raises("lever-arm fewer than two samples", "Accel lever-arm validation", lambda: add_invalid(timestamps=[0.0], acc=np.zeros((1, 3)), gyr=np.zeros((1, 3))), "at least two")
    expect_raises("lever-arm non-increasing sensor timestamps", "Accel lever-arm validation", lambda: add_invalid(timestamps=[-0.1, -0.1, 0.1]), "strictly increasing")
    expect_raises("lever-arm invalid accelerometer shape", "Accel lever-arm validation", lambda: add_invalid(acc=np.zeros((3, 2))), "shape (N, 3)")
    expect_raises("lever-arm invalid gyroscope shape", "Accel lever-arm validation", lambda: add_invalid(gyr=np.zeros((3, 2))), "shape (N, 3)")
    expect_raises("lever-arm shifted query outside support", "Accel lever-arm validation", lambda: add_invalid(tau=1.0).chi2(), "outside sample support")

    def static_limit():
        case = _accel_lever_case(lever_arm=np.array([0.4, -0.1, 0.2]), omega_at_query=np.zeros(3), alpha_body=np.zeros(3), force_slope=np.zeros(3), acceleration_world=np.zeros(3))
        graph_complex, _, _, _ = make_accel_lever_arm_graph(case)
        complex_chi = float(graph_complex.chi2())
        graph_simple = mrob.FGraph()
        pose = graph_simple.add_node_pose_3d(case["poses"][1], mrob.NODE_ANCHOR)
        nC = graph_simple.add_node_pose_3d(case["T_B_I"], mrob.NODE_ANCHOR)
        nt = graph_simple.add_node_scalar(case["tau_truth"], mrob.NODE_ANCHOR)
        graph_simple.add_factor_accel_gravity_calib(case["pose_time"], case["timestamps"], case["accelerometer"], case["gravity"], pose, nC, nt, np.eye(3))
        simple_chi = float(graph_simple.chi2())
        assert complex_chi < 1e-18 and simple_chi < 1e-18
        return f"complex={complex_chi:.3e}, simple={simple_chi:.3e}"

    def translation_sensitivity():
        graph_truth, _, _, case = make_accel_lever_arm_graph()
        graph_shift, _, _, _ = make_accel_lever_arm_graph(case, lever_arm_override=case["lever_arm"] + np.array([0.2, -0.1, 0.15]))
        simple_a, _, _, _ = make_accel_graph(np.zeros(3))
        simple_b, _, _, _ = make_accel_graph(np.array([2.0, -1.0, 0.5]))
        assert float(graph_truth.chi2()) < 1e-18
        assert float(graph_shift.chi2()) > 1e-5
        assert abs(float(simple_a.chi2()) - float(simple_b.chi2())) < 1e-20
        return f"complex shifted chi2={float(graph_shift.chi2()):.3e}; simple delta={abs(float(simple_a.chi2())-float(simple_b.chi2())):.3e}"

    def centripetal_term():
        case = _accel_lever_case(omega_at_query=np.array([0.45, -0.25, 0.15]), alpha_body=np.zeros(3), force_slope=np.zeros(3))
        truth, _, _, _ = make_accel_lever_arm_graph(case)
        wrong, _, _, _ = make_accel_lever_arm_graph(case, lever_arm_override=np.zeros(3))
        assert float(truth.chi2()) < 1e-18
        assert float(wrong.chi2()) > 1e-5
        return f"truth={float(truth.chi2()):.3e}, zero-lever={float(wrong.chi2()):.3e}"

    def tangential_term():
        case = _accel_lever_case(omega_at_query=np.zeros(3), alpha_body=np.array([0.55, -0.3, 0.22]), force_slope=np.zeros(3))
        truth, _, _, _ = make_accel_lever_arm_graph(case)
        wrong, _, _, _ = make_accel_lever_arm_graph(case, lever_arm_override=np.zeros(3))
        assert float(truth.chi2()) < 1e-18
        assert float(wrong.chi2()) > 1e-5
        return f"truth={float(truth.chi2()):.3e}, zero-lever={float(wrong.chi2()):.3e}"

    def bias_sensitivity():
        truth, _, _, case = make_accel_lever_arm_graph()
        wrong, _, _, _ = make_accel_lever_arm_graph(case, bias_override=case["bias"] + np.array([0.03, -0.02, 0.01]))
        assert float(truth.chi2()) < 1e-18
        assert float(wrong.chi2()) > 1e-7
        return f"wrong-bias chi2={float(wrong.chi2()):.3e}"

    def temporal_sensitivity():
        truth, _, _, case = make_accel_lever_arm_graph()
        wrong, _, _, _ = make_accel_lever_arm_graph(case, tau_initial=case["tau_truth"] + 0.02)
        assert float(truth.chi2()) < 1e-18
        assert float(wrong.chi2()) > 1e-6
        return f"truth={float(truth.chi2()):.3e}, shifted tau={float(wrong.chi2()):.3e}"

    def isolated_complex_accel_convergence():
        gravity = np.array([0.0, 0.0, -9.81])
        C_truth = mrob.SO3(np.array([0.12, -0.08, 0.06])).R()
        lever_truth = np.array([0.35, -0.22, 0.18])
        bias_truth = np.array([0.02, -0.015, 0.01])
        tau_truth = 0.035
        alpha_body = np.array([0.12, 0.07, -0.05])

        def trajectory_position(time):
            return np.array([0.3 * time + 0.2 * np.sin(1.3 * time), -0.15 * time + 0.1 * np.cos(0.9 * time), 0.08 * np.sin(1.7 * time)])

        def trajectory_rotation(time):
            return mrob.SO3(np.array([0.25 * np.sin(0.8 * time), -0.18 * np.cos(0.5 * time), 0.35 * np.sin(0.4 * time)])).R()

        def omega_body_time(time):
            return np.array([0.25 + alpha_body[0] * time, -0.18 + alpha_body[1] * time, 0.15 + alpha_body[2] * time])

        pose_times = np.linspace(0.0, 4.0, 17)
        pose_states = [se3_rt(trajectory_rotation(float(t)), trajectory_position(float(t))) for t in pose_times]
        query_times = []
        predicted_forces = []
        for pose_index in range(1, len(pose_times) - 1):
            t_prev, t_pose, t_next = pose_times[pose_index - 1], pose_times[pose_index], pose_times[pose_index + 1]
            h_prev = t_pose - t_prev
            h_next = t_next - t_pose
            c_prev = 2.0 / (h_prev * (h_prev + h_next))
            c_pose = -2.0 / (h_prev * h_next)
            c_next = 2.0 / (h_next * (h_prev + h_next))
            linear_acceleration_world = c_prev * trajectory_position(t_prev) + c_pose * trajectory_position(t_pose) + c_next * trajectory_position(t_next)
            omega = omega_body_time(t_pose + tau_truth)
            lever_acceleration = np.cross(alpha_body, lever_truth) + np.cross(omega, np.cross(omega, lever_truth))
            body_specific_force = trajectory_rotation(t_pose).T @ (gravity - linear_acceleration_world)
            query_times.append(float(t_pose + tau_truth))
            predicted_forces.append(C_truth.T @ (body_specific_force - lever_acceleration))

        query_times = np.asarray(query_times)
        predicted_forces = np.asarray(predicted_forces)
        sensor_timestamps = np.unique(np.concatenate(([query_times[0] - 0.5], query_times, [query_times[-1] + 0.5])))
        accelerometer = np.column_stack([np.interp(sensor_timestamps, query_times, predicted_forces[:, axis], left=predicted_forces[0, axis], right=predicted_forces[-1, axis]) for axis in range(3)])
        gyroscope = np.vstack([C_truth.T @ omega_body_time(float(t)) + bias_truth for t in sensor_timestamps])

        graph = mrob.FGraph()
        pose_nodes = [graph.add_node_pose_3d(pose, mrob.NODE_ANCHOR) for pose in pose_states]
        nC = graph.add_node_pose_3d(se3_rt(mrob.SO3(np.array([0.15, -0.10, 0.08])).R(), lever_truth + np.array([0.08, -0.05, 0.04])), mrob.NODE_STANDARD)
        nb = graph.add_node_landmark_3d(bias_truth + np.array([0.01, -0.008, 0.006]), mrob.NODE_STANDARD)
        nt = graph.add_node_scalar(tau_truth + 0.01, mrob.NODE_STANDARD)
        before_T = state(graph, nC).copy()
        before_bias = state(graph, nb).ravel().copy()
        before_tau = float(state(graph, nt)[0, 0])
        for pose_index in range(1, len(pose_times) - 1):
            graph.add_factor_accel_lever_arm_calib(float(pose_times[pose_index - 1]), float(pose_times[pose_index]), float(pose_times[pose_index + 1]), sensor_timestamps, accelerometer, gyroscope, gravity, pose_nodes[pose_index - 1], pose_nodes[pose_index], pose_nodes[pose_index + 1], nC, nb, nt, 10.0 * np.eye(3))
        before = float(graph.chi2())
        graph.solve(mrob.LM, 100, 1e-4, 1e-9, False)
        after = float(graph.chi2())
        after_T = state(graph, nC)
        after_bias = state(graph, nb).ravel()
        after_tau = float(state(graph, nt)[0, 0])
        assert before > after * 1e8
        assert np.linalg.norm(mrob.SE3(after_T).Ln()[:3] - np.array([0.12, -0.08, 0.06])) < 5e-4
        assert np.linalg.norm(after_T[:3, 3] - lever_truth) < 5e-3
        assert np.linalg.norm(after_bias - bias_truth) < 5e-4
        assert abs(after_tau - tau_truth) < 5e-4
        assert_finite_states(graph)
        return f"chi2 {before:.3e}->{after:.3e}; T_B_I moved {np.linalg.norm(after_T - before_T):.3e}, bias moved {np.linalg.norm(after_bias - before_bias):.3e}, tau {before_tau:.3f}->{after_tau:.3f}"

    def solve_with_gyro():
        graph, nodes, _, case = make_accel_lever_arm_graph(anchor_calibration=False)
        pose_nodes = nodes[:3]
        nC, nb, nt = nodes[3:]
        graph.add_factor_gyro_calib_prop(float(case["pose_times"][0]), float(case["pose_times"][1]), case["timestamps"], case["gyroscope"], pose_nodes[0], pose_nodes[1], nC, nb, nt, 0.1 * np.eye(3))
        before = float(graph.chi2())
        graph.solve(mrob.LM, 5)
        after = float(graph.chi2())
        assert np.isfinite(before) and np.isfinite(after)
        assert_finite_states(graph)
        assert graph.number_factors() == 2
        return f"chi2={before:.3e}->{after:.3e}"

    def coexist_full_graph():
        graph, nodes, _, case = make_accel_lever_arm_graph(anchor_calibration=False)
        pose_nodes = nodes[:3]
        nC, nb, nt = nodes[3:]
        nL = graph.add_node_pose_3d(se3([0.08, -0.04, 0.06], [0.3, -0.15, 0.2]), mrob.NODE_ANCHOR)
        nTauL = graph.add_node_scalar(0.0, mrob.NODE_ANCHOR)
        lidar_times = np.linspace(float(case["pose_times"][0]) - 0.1, float(case["pose_times"][2]) + 0.1, 9)
        lidar_poses = [body_pose(float(t)) * se3([0.08, -0.04, 0.06], [0.3, -0.15, 0.2]) for t in lidar_times]
        graph.add_factor_gyro_calib_prop(float(case["pose_times"][0]), float(case["pose_times"][1]), case["timestamps"], case["gyroscope"], pose_nodes[0], pose_nodes[1], nC, nb, nt, 0.1 * np.eye(3))
        graph.add_factor_lidar_calib_odometry(float(case["pose_times"][0]), float(case["pose_times"][2]), lidar_times, lidar_poses, pose_nodes[0], pose_nodes[2], nL, nTauL, 1e-3 * np.eye(6))
        before_log = print_graph_to_log(graph, "accel_lever_full_before")
        before = float(graph.chi2())
        graph.solve(mrob.LM, 3)
        after_log = print_graph_to_log(graph, "accel_lever_full_after")
        after = float(graph.chi2())
        assert np.isfinite(before) and np.isfinite(after)
        assert_finite_states(graph)
        assert graph.number_factors() == 3
        return f"chi2={before:.3e}->{after:.3e}; logs={before_log.name},{after_log.name}"

    check("static limiting case matches simple accelerometer", "Accel lever-arm", static_limit, "sign or frame-convention mismatch")
    check("T_B_I translation affects complex factor only", "Accel lever-arm", translation_sensitivity, "implementation bug")
    check("centripetal omega-cross-omega lever term", "Accel lever-arm", centripetal_term, "implementation bug")
    check("tangential alpha-cross-lever term", "Accel lever-arm", tangential_term, "implementation bug")
    check("gyroscope bias changes lever-arm objective", "Accel lever-arm", bias_sensitivity, "implementation bug")
    check("tau_I changes shifted accelerometer/gyro lookup", "Accel lever-arm", temporal_sensitivity, "implementation bug")
    check("isolated complex accelerometer graph convergence", "Accel lever-arm", isolated_complex_accel_convergence, "insufficient excitation or unobservable variable")
    check("coexists with FactorGyroCalibProp", "Accel lever-arm", solve_with_gyro, "node ordering or solver integration bug")
    check("coexists with gyro and LiDAR factors", "Accel lever-arm", coexist_full_graph, "node ordering or solver integration bug")
    record(
        "all 28 local-coordinate Jacobian blocks",
        "Accel lever-arm",
        "LIMITED",
        "Current Python FGraph binding exposes graph chi-square but not per-factor residual/Jacobian blocks.",
        "current binding surface limitation",
    )

def final_summary():
    summary = pd.DataFrame(RESULTS)
    passed = int((summary["status"] == "PASS").sum())
    failed = int((summary["status"] == "FAIL").sum())
    limited = int((summary["status"] == "LIMITED").sum())
    report = {
        "build_command_used": BUILD_COMMAND_USED,
        "tests_passed": passed,
        "tests_failed": failed,
        "limited_checks": limited,
        "implementation_bugs_found": IMPLEMENTATION_BUGS_FOUND,
        "binding_mismatches_found": BINDING_MISMATCHES_FOUND,
        "impossible_without_additional_read_only_binding": IMPOSSIBLE_WITHOUT_BINDING,
    }
    print("Final report:")
    for key, value in report.items():
        print(f"- {key}: {value}")
    assert failed == 0, "One or more validation checks failed; inspect the summary table."
    return summary, report
