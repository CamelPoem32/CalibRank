import numpy as np


def test_so3_wrappers_prefer_mrob_when_available(monkeypatch):
    from src.calib_observability import lie_so3

    class FakeSO3:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=float)

        def R(self):
            return np.full((3, 3), 2.0)

        def Ln(self):
            return np.array([0.4, 0.5, 0.6])

    class FakeMrob:
        SO3 = FakeSO3

        @staticmethod
        def hat3(omega):
            return np.full((3, 3), 3.0) + float(np.asarray(omega)[0])

        @staticmethod
        def left_jacobian_SO3(omega):
            return np.eye(3) * 4.0

        @staticmethod
        def inv_left_jacobian_SO3(omega):
            return np.eye(3) * 5.0

    monkeypatch.setattr(lie_so3, "_mrob", FakeMrob)
    monkeypatch.setattr(lie_so3, "_mrob_geometry_is_usable", lambda group_name="SO3": True)

    assert np.allclose(lie_so3.so3_hat([1.0, 2.0, 3.0]), np.full((3, 3), 4.0))
    assert np.allclose(lie_so3.so3_exp([0.1, 0.2, 0.3]), np.full((3, 3), 2.0))
    assert np.allclose(lie_so3.so3_log(np.eye(3)), np.array([0.4, 0.5, 0.6]))
    assert np.allclose(lie_so3.so3_left_jacobian([0.1, 0.2, 0.3]), np.eye(3) * 4.0)
    assert np.allclose(lie_so3.so3_left_jacobian_inverse([0.1, 0.2, 0.3]), np.eye(3) * 5.0)


def test_se3_wrappers_prefer_mrob_when_available(monkeypatch):
    from src.calib_observability import lie_se3

    class FakeSE3:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=float)

        def T(self):
            return np.eye(4) * 2.0

        def Ln(self):
            return np.arange(6, dtype=float)

        def inv(self):
            return FakeSE3(np.eye(4) * 3.0)

        def adj(self):
            return np.eye(6) * 4.0

        def transform(self, point):
            return np.asarray(point, dtype=float) + 10.0

    class FakeMrob:
        SE3 = FakeSE3

        @staticmethod
        def hat6(xi):
            return np.eye(4) * 6.0

        @staticmethod
        def curley_wedge(xi):
            return np.eye(6) * 7.0

    monkeypatch.setattr(lie_se3, "_mrob", FakeMrob)
    monkeypatch.setattr(lie_se3, "_mrob_geometry_is_usable", lambda group_name="SE3": True)

    assert np.allclose(lie_se3.se3_hat(np.arange(6)), np.eye(4) * 6.0)
    assert np.allclose(lie_se3.se3_exp(np.arange(6)), np.eye(4) * 2.0)
    assert np.allclose(lie_se3.se3_log(np.eye(4)), np.arange(6, dtype=float))
    assert np.allclose(lie_se3.se3_inverse(np.eye(4)), np.eye(4) * 2.0)
    assert np.allclose(lie_se3.se3_adjoint(np.eye(4)), np.eye(6) * 4.0)
    assert np.allclose(lie_se3.se3_little_adjoint(np.arange(6)), np.eye(6) * 7.0)
    assert np.allclose(lie_se3.transform_point(np.eye(4), [1.0, 2.0, 3.0]), [11.0, 12.0, 13.0])


def test_se2_wrappers_use_mrob_se3_embedding_when_available(monkeypatch):
    from src.calib_observability import lie_se2

    class FakeSE3:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=float)

        def T(self):
            T = np.eye(4)
            T[0, 0] = 0.0
            T[0, 1] = -1.0
            T[1, 0] = 1.0
            T[1, 1] = 0.0
            T[0, 3] = 2.0
            T[1, 3] = 3.0
            return T

        def Ln(self):
            return np.array([0.0, 0.0, 0.25, 4.0, 5.0, 0.0])

        def inv(self):
            return self

        def adj(self):
            return np.eye(6)

    class FakeMrob:
        SE3 = FakeSE3

    monkeypatch.setattr(lie_se2, "_mrob", FakeMrob)
    monkeypatch.setattr(lie_se2, "_mrob_geometry_is_usable", lambda group_name="SE3": True)

    assert np.allclose(lie_se2.se2_exp([0.1, 1.0, 2.0]), [[0.0, -1.0, 2.0], [1.0, 0.0, 3.0], [0.0, 0.0, 1.0]])
    assert np.allclose(lie_se2.se2_log(np.eye(3)), [0.25, 4.0, 5.0])
    assert lie_se2.se2_inverse(np.eye(3)).shape == (3, 3)
    assert lie_se2.se2_adjoint(np.eye(3)).shape == (3, 3)
