"""
Module for Pytorch dataset representations
"""

import torch
from torch.utils.data import Dataset

class SlicesDataset(Dataset):
    """
    This class represents an indexable Torch dataset
    which could be consumed by the PyTorch DataLoader class
    """
    def __init__(self, data):
        self.data = data

        self.slices = []

        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))

    def __getitem__(self, idx):
        """
        Este método é chamado pelo PyTorch DataLoader para retornar uma amostra.

        Arguments: 
            idx {int} -- id da amostra

        Returns:
            sample {dict} -- contém:
                - "id": índice global da fatia
                - "image": Tensor [1, W, H] da imagem MRI
                - "seg": Tensor [1, W, H] da segmentação
        """
        slc = self.slices[idx]   # (id_volume, id_slice)
        sample = dict()
        sample["id"] = idx

        vol_id, slice_id = slc

        # Pega a fatia correspondente (numpy array)
        image_slice = self.data[vol_id]["image"][slice_id, :, :]
        seg_slice   = self.data[vol_id]["seg"][slice_id, :, :]

        # Adiciona dimensão [1, W, H] e converte para tensor float32/long
        sample["image"] = torch.tensor(image_slice[None, :, :], dtype=torch.float32)
        sample["seg"]   = torch.tensor(seg_slice[None, :, :], dtype=torch.long)

        return sample

    def __len__(self):
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            int
        """
        return len(self.slices)
