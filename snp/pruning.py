import torch
from torch import nn

import numpy as np

import heapq
import math

from new_code import utils


def compute_pruned_nodes(local_scores, calibrated_scores, ratio):
    def simulate_dispensing():
        n_dispensers = len(local_scores)
        stacks = []
        for i in range(n_dispensers):
            a, b = local_scores[i], calibrated_scores[i]
            order = torch.argsort(a, descending=True)
            stacks.append(torch.stack([a[order], b[order]], dim=1).tolist())

        heap = []
        for i, s in enumerate(stacks):
            a_val, b_val = s[-1]
            heapq.heappush(heap, (b_val, i, a_val))

        dispensed = []
        while heap:
            b_val, i, a_val = heapq.heappop(heap)
            stacks[i].pop()
            dispensed.append((i, a_val, b_val))
            if stacks[i]:
                new_a, new_b = stacks[i][-1]
                heapq.heappush(heap, (new_b, i, new_a))

        return dispensed

    def dispensed_counts_for_percentage(dispensed, percentage, n_dispensers):
        total = len(dispensed)
        n_to_pop = math.ceil(percentage * total)
        counts = [0] * n_dispensers
        for i, _, _ in dispensed[:n_to_pop]:
            counts[i] += 1
        return n_to_pop, counts

    dispensed = simulate_dispensing()

    n_popped, counts = dispensed_counts_for_percentage(
        dispensed, ratio, len(local_scores)
    )

    units_pruned_per_group = [
        torch.argsort(scores, descending=False)[:c]
        for c, scores in zip(counts, local_scores)
    ]

    return units_pruned_per_group, n_popped


def lls_reconstruction(W, C, n_in, k, order, n_removed, uses_bias):
    i = int(uses_bias)
    order = torch.concat([torch.arange(i, device=C.device), i + order])
    undo_order = torch.argsort(order, descending=False)

    C = C[order][:, order]

    M_inv = torch.linalg.cholesky(C)
    M = torch.linalg.solve(
        M_inv, torch.eye(len(M_inv), device=M_inv.device, dtype=M_inv.dtype)
    )

    n_kept_params = len(C) - (n_removed * k)

    M_inv[:, n_kept_params:] = 0.0
    M[:, n_kept_params:] = 0.0
    M[n_kept_params:] = 0.0

    A = M_inv @ M
    A = A[undo_order][:, undo_order]

    return W @ A


def slice_layer(layer, idcs_removed, order, use_bias, eps, reconstruct=True):
    n_in, k, uses_bias, alive, order, W, A, _, _ = utils.prepare_vgg_layer(
        layer, order, use_bias, eps
    )

    alive_idcs_removed = idcs_removed[torch.isin(idcs_removed, torch.where(alive)[0])]

    if reconstruct:
        W = lls_reconstruction(
            W,
            A,
            n_in,
            k,
            order,
            n_removed=len(alive_idcs_removed),
            uses_bias=uses_bias,
        )
    else:
        W_ = W[int(use_bias) :]
        alive_idcs_removed_ = utils.block_expand(alive_idcs_removed, k)
        W_[:, alive_idcs_removed_] = 0.0
        W[int(use_bias) :] = W_

    if layer.bias is not None and uses_bias:
        b = W[:, 0].to(dtype=layer.bias.dtype, device=layer.bias.device)
        layer.bias.data = b

        W = W[:, 1:]

    W = W.to(dtype=layer.weight.dtype, device=layer.weight.device)

    original_W_shape = layer.weight.shape
    W_ = layer.weight.data.reshape(original_W_shape[0], np.prod(original_W_shape[1:]))
    W_[:, alive] = W

    layer.weight.data = W_.reshape(original_W_shape)


def prune_group(
    group, input_pruned_layers, idcs_removed, order, use_bias, eps, reconstruct
):
    for layer in input_pruned_layers:
        slice_layer(layer, idcs_removed, order, use_bias, eps, reconstruct)

    group.prune(idcs_removed.tolist())


def pruning_groups(DG, ignored_layers):
    for group in DG.get_all_groups(
        ignored_layers=ignored_layers, root_module_types=[nn.Conv2d, nn.Linear]
    ):
        yield group, list(
            map(
                lambda g: g[0].target.module,
                filter(
                    lambda g: isinstance(g[0].target.module, (nn.Linear, nn.Conv2d))
                    and g[0].handler.__name__ == "prune_in_channels",
                    group,
                ),
            )
        )


def prune_model(
    DG,
    ignored_layers,
    units_pruned_per_group,
    unit_orders_per_group,
    use_bias,
    eps,
    reconstruct,
):
    for (group, input_pruned_layers), idcs_removed, order in zip(
        pruning_groups(DG, ignored_layers),
        units_pruned_per_group,
        unit_orders_per_group,
    ):
        prune_group(
            group, input_pruned_layers, idcs_removed, order, use_bias, eps, reconstruct
        )
