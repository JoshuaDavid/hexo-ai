"""Tests for D6 symmetry group on 32x32 torus."""

import numpy as np
import torch
from hexo.game.constants import BOARD_SIZE, HEX_DIRECTIONS
from hexo.encoding.symmetry import (
    SYMMETRY_COEFFS, PERMS, INV_PERMS, PERMS_TORCH,
    apply_symmetry_planes, N,
)


NN = N * N


class TestSymmetryBijection:
    def test_each_permutation_is_bijection(self):
        for k in range(12):
            assert len(set(PERMS[k])) == NN, f"Symmetry {k}: not a bijection"

    def test_inverse_correctness(self):
        for k in range(12):
            for i in range(NN):
                assert INV_PERMS[k, PERMS[k, i]] == i, \
                    f"Symmetry {k}: inverse failed at {i}"

    def test_identity_is_identity(self):
        for i in range(NN):
            assert PERMS[0, i] == i


class TestSymmetryDirectionPreservation:
    def test_hex_directions_preserved(self):
        """Each symmetry maps the set of hex directions to itself (up to sign)."""
        for k in range(12):
            a, b, c, d = SYMMETRY_COEFFS[k]
            transformed = set()
            for dq, dr in HEX_DIRECTIONS:
                new_dq = (a * dq + b * dr) % N
                new_dr = (c * dq + d * dr) % N
                # Normalize: direction and its negative are the same line
                if new_dq > N // 2:
                    new_dq = N - new_dq
                    new_dr = N - new_dr
                if new_dq == 0 and new_dr > N // 2:
                    new_dr = N - new_dr
                transformed.add((new_dq, new_dr % N))

            original = set()
            for dq, dr in HEX_DIRECTIONS:
                ndq = dq % N
                ndr = dr % N
                if ndq > N // 2:
                    ndq = N - ndq
                    ndr = N - ndr
                if ndq == 0 and ndr > N // 2:
                    ndr = N - ndr
                original.add((ndq, ndr % N))

            assert transformed == original, \
                f"Symmetry {k}: directions not preserved"


class TestSymmetryGroupClosure:
    def test_closure(self):
        """Composing any two symmetries gives another in the group."""
        for i in range(12):
            for j in range(12):
                composed = PERMS[i][PERMS[j]]
                found = any(np.array_equal(composed, PERMS[k]) for k in range(12))
                assert found, f"Symmetry {i} o {j} not in group"


class TestApplySymmetryPlanes:
    def test_identity_noop(self):
        planes = torch.randn(3, N, N)
        result = apply_symmetry_planes(planes, 0)
        assert torch.allclose(result, planes)

    def test_roundtrip(self):
        """Applying symmetry then inverse gives back original."""
        planes = torch.randn(3, N, N)
        for k in range(12):
            transformed = apply_symmetry_planes(planes, k)
            # Find inverse: for each k, the inverse is the k' where PERMS[k'][PERMS[k]] = identity
            inv_k = None
            for j in range(12):
                if np.array_equal(PERMS[j][PERMS[k]], np.arange(NN)):
                    inv_k = j
                    break
            assert inv_k is not None
            restored = apply_symmetry_planes(transformed, inv_k)
            assert torch.allclose(restored, planes, atol=1e-6), \
                f"Roundtrip failed for symmetry {k}"

    def test_stone_moves_correctly(self):
        """A single stone at (q,r) moves to the correct transformed position."""
        planes = torch.zeros(2, N, N)
        q, r = 5, 10
        planes[0, q, r] = 1.0

        # Apply 180 deg rotation (k=3): new_q = -q % N, new_r = -r % N
        result = apply_symmetry_planes(planes, 3)
        new_q = (-q) % N
        new_r = (-r) % N
        assert result[0, new_q, new_r] == 1.0
        assert result[0].sum() == 1.0
