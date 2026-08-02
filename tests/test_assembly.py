from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.calib_observability.assembly import JacobianBlock, VariableLayout, assemble_jacobian_dense, assemble_jacobian_sparse, make_residual_blocks


def test_dense_sparse_assembly_match() -> None:
    layout = VariableLayout.from_specs([("T_W_B_0", 6, "trajectory"), ("T_B_L", 6, "calibration"), ("tau_L", 1, "calibration")])
    residual_blocks = make_residual_blocks([("r0", 3, np.eye(3), "measurement"), ("p0", 1, np.eye(1), "prior")])
    blocks = [
        JacobianBlock("r0", "T_W_B_0", np.ones((3, 6))),
        JacobianBlock("r0", "T_B_L", 2.0 * np.ones((3, 6))),
        JacobianBlock("p0", "tau_L", np.ones((1, 1))),
    ]
    bd = assemble_jacobian_dense(layout, residual_blocks, blocks)
    bs = assemble_jacobian_sparse(layout, residual_blocks, blocks)
    assert np.allclose(bd.J, bs.J.toarray())
    assert bd.J_T.shape == (4, 6)
    assert bd.J_C.shape == (4, 7)
