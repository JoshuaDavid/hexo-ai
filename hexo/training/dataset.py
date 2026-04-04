"""Training dataset with D6 augmentation."""

import random

import torch
from torch.utils.data import Dataset

from hexo.game.constants import BOARD_SIZE, N_CELLS
from hexo.encoding.symmetry import apply_symmetry_planes, PERMS


class SelfPlayDataset(Dataset):
    """Dataset of self-play positions with D6 augmentation.

    Visit counts are stored sparsely and densified on the fly.
    A random D6 symmetry is applied at access time.
    """

    def __init__(self, planes_list, visit_dicts_list, values_list,
                 round_ids_list, current_round, decay=0.75, augment=True):
        """
        Args:
            planes_list: list of [3, 32, 32] tensors
            visit_dicts_list: list of {cell_idx: prob} dicts
            values_list: list of float value targets
            round_ids_list: list of int round IDs
            current_round: current round for age weighting
            decay: per-round weight decay
            augment: whether to apply D6 augmentation
        """
        self.planes = planes_list
        self.visit_dicts = visit_dicts_list
        self.values = values_list
        self.augment = augment

        ages = torch.tensor([current_round - r for r in round_ids_list],
                            dtype=torch.float32)
        self.weights = decay ** ages

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        planes = self.planes[idx]
        value = self.values[idx]
        visit_dict = self.visit_dicts[idx]

        k = random.randint(0, 11) if self.augment else 0

        if k != 0:
            planes = apply_symmetry_planes(planes, k)

        # Build dense visit vector with symmetry
        visit_vec = torch.zeros(N_CELLS)
        if visit_dict:
            if k != 0:
                perm = PERMS[k]
                for cell_idx, prob in visit_dict.items():
                    new_idx = int(perm[cell_idx])
                    visit_vec[new_idx] = prob
            else:
                for cell_idx, prob in visit_dict.items():
                    visit_vec[cell_idx] = prob

        return planes, visit_vec, torch.tensor(value, dtype=torch.float32)

    def get_sample_weights(self):
        """Return per-sample weights for WeightedRandomSampler."""
        return self.weights
