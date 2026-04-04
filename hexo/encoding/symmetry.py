"""D6 symmetry group for the hex grid on a 32x32 torus.

Precomputes 12 permutation tables (6 rotations x 2 reflections) for
axial hex coordinates (q, r) mod 32. Used to randomly augment training
samples so the model sees each position in all orientations.
"""

import numpy as np
import torch

from hexo.game.constants import BOARD_SIZE, HEX_DIRECTIONS

N = BOARD_SIZE  # 32

# 12 symmetry transforms as linear coefficient matrices (a, b, c, d):
#   new_q = (a*q + b*r) % N
#   new_r = (c*q + d*r) % N
SYMMETRY_COEFFS = [
    # 6 rotations
    ( 1,  0,  0,  1),   # R0: identity
    ( 0, -1,  1,  1),   # R1: 60 deg
    (-1, -1,  1,  0),   # R2: 120 deg
    (-1,  0,  0, -1),   # R3: 180 deg
    ( 0,  1, -1, -1),   # R4: 240 deg
    ( 1,  1, -1,  0),   # R5: 300 deg
    # 6 reflections (apply (q,r)->(r,q) then rotate)
    ( 0,  1,  1,  0),   # S0: reflect
    (-1,  0,  1,  1),   # S1: reflect + R1
    (-1, -1,  0,  1),   # S2: reflect + R2
    ( 0, -1, -1,  0),   # S3: reflect + R3
    ( 1,  0, -1, -1),   # S4: reflect + R4
    ( 1,  1,  0, -1),   # S5: reflect + R5
]


def _build_permutations():
    """Build forward permutation tables: PERMS[k][old_flat] = new_flat."""
    perms = np.zeros((12, N * N), dtype=np.int64)
    for k, (a, b, c, d) in enumerate(SYMMETRY_COEFFS):
        for q in range(N):
            for r in range(N):
                old_idx = q * N + r
                new_q = (a * q + b * r) % N
                new_r = (c * q + d * r) % N
                new_idx = new_q * N + new_r
                perms[k, old_idx] = new_idx
    return perms


PERMS = _build_permutations()                          # [12, 1024]
INV_PERMS = np.zeros_like(PERMS)                       # [12, 1024]
for _k in range(12):
    for _i in range(N * N):
        INV_PERMS[_k, PERMS[_k, _i]] = _i

PERMS_TORCH = torch.from_numpy(PERMS).long()           # [12, 1024]
INV_PERMS_TORCH = torch.from_numpy(INV_PERMS).long()   # [12, 1024]


def apply_symmetry_planes(planes: torch.Tensor, k: int) -> torch.Tensor:
    """Apply symmetry k to board planes [C, N, N]. Returns new planes."""
    C = planes.shape[0]
    flat = planes.reshape(C, -1)       # [C, 1024]
    inv = INV_PERMS_TORCH[k]          # [1024]
    return flat[:, inv].reshape(C, N, N)


def apply_symmetry_flat(flat_indices: torch.Tensor, k: int) -> torch.Tensor:
    """Remap flat cell indices under symmetry k."""
    if k == 0:
        return flat_indices
    perm = PERMS_TORCH[k]
    return perm[flat_indices]
