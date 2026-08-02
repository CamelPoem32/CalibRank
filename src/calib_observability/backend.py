"""Pose-provider interfaces for future factor-graph integration."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from .lie_se3 import se3_adjoint, se3_inverse, se3_log
from numpy.typing import ArrayLike, NDArray


class PoseProvider(Protocol):
    """Protocol supplying estimated body poses and twists at requested times."""

    def poses_at(self, times: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return `T_W_B(t)`, shape `(N, 4, 4)`, using left-perturbation convention."""

    def body_twists_at(self, times: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return body twists, shape `(N, 6)`, in rotation-first ordering."""

    def spatial_twists_at(self, times: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return spatial twists, shape `(N, 6)`, in rotation-first ordering."""

    def poses_and_twists_at(self, times: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return poses, body twists, and spatial twists for times `(N,)`."""


class TruePoseProvider:
    """Dummy backend returning exact simulated true poses and local twists.

    Twists are evaluated by a small centered SE(3) geodesic difference around
    each requested time. The tangent ordering is rotation-first and compatible
    with left perturbations. This is a deterministic interpolation/linearization
    helper for notebooks, not an optimized estimator backend.
    """

    def __init__(self, pose_function: object, *, twist_step: float = 1e-7):
        self.pose_function = pose_function
        self.twist_step = float(twist_step)

    def poses_at(self, times: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the stored true pose function at each time."""

        t = np.asarray(times, dtype=float)
        if t.ndim != 1 or not np.all(np.isfinite(t)):
            raise ValueError("times must be a finite vector")
        poses = [np.asarray(self.pose_function(float(ti)), dtype=float) for ti in t]
        out = np.stack(poses, axis=0) if poses else np.zeros((0, 4, 4))
        if out.shape != (t.size, 4, 4):
            raise ValueError("pose_function must return transforms of shape (4, 4)")
        return out

    def body_twists_at(self, times: ArrayLike) -> NDArray[np.float64]:
        """Return centered finite-difference body twists, shape `(N, 6)`."""

        t = np.asarray(times, dtype=float).reshape(-1)
        twists = []
        for query_time in t:
            dt = self.twist_step
            earlier_pose = np.asarray(self.pose_function(float(query_time - dt)), dtype=float)
            later_pose = np.asarray(self.pose_function(float(query_time + dt)), dtype=float)
            # inv(T_minus): (4, 4), T_plus: (4, 4) -> relative motion over 2 dt.
            twists.append(se3_log(se3_inverse(earlier_pose) @ later_pose) / (2.0 * dt))
        return np.vstack(twists) if twists else np.zeros((0, 6))

    def spatial_twists_at(self, times: ArrayLike) -> NDArray[np.float64]:
        """Return spatial twists, shape `(N, 6)`, via `Adj(T) @ xi_body`."""

        poses = self.poses_at(times)
        body_twists = self.body_twists_at(times)
        return np.vstack([se3_adjoint(pose) @ twist for pose, twist in zip(poses, body_twists)]) if body_twists.size else np.zeros((0, 6))

    def poses_and_twists_at(self, times: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return poses `(N,4,4)`, body twists `(N,6)`, and spatial twists `(N,6)`."""

        poses = self.poses_at(times)
        body_twists = self.body_twists_at(times)
        spatial_twists = np.vstack([se3_adjoint(pose) @ twist for pose, twist in zip(poses, body_twists)]) if body_twists.size else np.zeros((0, 6))
        return poses, body_twists, spatial_twists


class MrobPoseProvider:
    """Placeholder for future MROB-backed pose estimates."""

    def __init__(self, graph: object | None = None):
        self.graph = graph

    def poses_at(self, times: ArrayLike) -> NDArray[np.float64]:
        """Future MROB pose query hook."""

        _ = times
        raise NotImplementedError(
            "Future MROB integration point: query optimized poses and convert "
            "their Jacobian/tangent conventions before observability assembly."
        )

    def body_twists_at(self, times: ArrayLike) -> NDArray[np.float64]:
        """Future MROB body-twist query hook."""

        _ = times
        raise NotImplementedError("Future MROB integration point: body twists are not implemented.")

    def spatial_twists_at(self, times: ArrayLike) -> NDArray[np.float64]:
        """Future MROB spatial-twist query hook."""

        _ = times
        raise NotImplementedError("Future MROB integration point: spatial twists are not implemented.")

    def poses_and_twists_at(self, times: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Future MROB combined pose/twist query hook."""

        return self.poses_at(times), self.body_twists_at(times), self.spatial_twists_at(times)


def estimate_poses_dummy(dataset: object) -> TruePoseProvider:
    """Return a dummy provider that uses the simulated true trajectory."""

    trajectory = getattr(dataset, "trajectory", dataset)
    pose_function = getattr(trajectory, "pose_at", None)
    if pose_function is None:
        raise ValueError("dataset or trajectory must expose pose_at(t)")
    return TruePoseProvider(pose_function)
