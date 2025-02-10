from typing import Any, List

import numpy as np
import torch

from torchsparse import SparseTensor

__all__ = ["sparse_collate", "sparse_collate_fn"]


def sparse_collate(inputs: List[SparseTensor]) -> SparseTensor:
    coords, feats = [], []
    stride = inputs[0].stride

    for k, x in enumerate(inputs):
        if isinstance(x.coords, np.ndarray):
            x.coords = torch.tensor(x.coords)
        if isinstance(x.feats, np.ndarray):
            x.feats = torch.tensor(x.feats)

        assert isinstance(x.coords, torch.Tensor), type(x.coords)
        assert isinstance(x.feats, torch.Tensor), type(x.feats)
        assert x.stride == stride, (x.stride, stride)

        input_size = x.coords.shape[0]
        batch = torch.full((input_size, 1), k, device=x.coords.device, dtype=torch.int)
        coords.append(torch.cat((batch, x.coords), dim=1))
        feats.append(x.feats)

    coords = torch.cat(coords, dim=0)
    feats = torch.cat(feats, dim=0)
    output = SparseTensor(coords=coords, feats=feats, stride=stride)
    return output


def sparse_collate_fn(inputs: List[Any]) -> Any:
    print(inputs[0])
    if isinstance(inputs[0], tuple): 
        coords_list, feats_list = zip(*inputs) # Sépare les coordonnées et features

        # Convertir en torch.Tensor si nécessaire
        coords_list = [torch.tensor(c, dtype=torch.float32,device='cuda') if isinstance(c, np.ndarray) else c for c in coords_list]
        feats_list = [torch.tensor(f, dtype=torch.float32,device='cuda') if isinstance(f, np.ndarray) else f for f in feats_list]

        # Ajouter un batch index aux coordonnées
        batch_indices = [torch.full((c.shape[0], 1), i, dtype=torch.int32,device='cuda') for i, c in enumerate(coords_list)]
        batched_coords = torch.cat([torch.cat((b, c), dim=1) for b, c in zip(batch_indices, coords_list)], dim=0)
        batched_feats = torch.cat(feats_list, dim=0)
        print(batched_coords.shape)
        # Retourne un SparseTensor batché
        return SparseTensor(coords=batched_coords, feats=batched_feats)

    elif isinstance(inputs[0], dict):
        output = {}
        for name in inputs[0].keys():
            if isinstance(inputs[0][name], dict):
                output[name] = sparse_collate_fn([input[name] for input in inputs])
            elif isinstance(inputs[0][name], np.ndarray):
                output[name] = torch.stack(
                    [torch.tensor(input[name]) for input in inputs], dim=0
                )
            elif isinstance(inputs[0][name], torch.Tensor):
                output[name] = torch.stack([input[name] for input in inputs], dim=0)
            elif isinstance(inputs[0][name], SparseTensor):
                output[name] = sparse_collate([input[name] for input in inputs])
            else:
                output[name] = [input[name] for input in inputs]
        return output
    else:
        return inputs
