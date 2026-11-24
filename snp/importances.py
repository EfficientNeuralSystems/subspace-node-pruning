import torch

from new_code import utils, pruning


def local_importances_taylor_second_order(W, A, G, n_in, k, uses_bias):
    A_inv = torch.linalg.solve(A, torch.eye(len(A), dtype=A.dtype, device=A.device))

    if uses_bias:
        b, W = W[:, :1], W[:, 1:]

        bGb = (b.T @ b) if G is None else (b.T @ G @ b)
        bias_importance = 1 / 2 * torch.diag(bGb) / torch.diag(A_inv[-1:, -1:])

        A_inv = A_inv[1:][:, 1:]
    else:
        bias_importance = torch.tensor([], dtype=W.dtype, device=W.device)

    A_inv = torch.einsum("iaib->iab", A_inv.reshape(n_in, k, n_in, k))
    A_inv = torch.linalg.inv(A_inv)

    WGW = W.T @ G @ W
    WGW = torch.einsum("iaib->iab", WGW.reshape(n_in, k, n_in, k))

    weight_importances = (
        torch.vmap(torch.trace)(torch.einsum("bij,bjk->bik", WGW, A_inv)) / 2
    )
    weight_importances = torch.repeat_interleave(weight_importances, k)

    return torch.concat([bias_importance, weight_importances])


def calibrated_importances_taylor_second_order(W, A, G):
    M_inv = torch.linalg.cholesky(A)
    return torch.diag(M_inv.T @ W.T @ G @ W @ M_inv) / 2


def local_importances_activity(A, n_in, k, uses_bias):
    A_inv = torch.linalg.solve(A, torch.eye(len(A), dtype=A.dtype, device=A.device))

    if uses_bias:
        bias_importance = 1 / (2 * torch.diag(A_inv[:1, :1]))
        A_inv = A_inv[1:][:, 1:]
    else:
        bias_importance = torch.tensor([], dtype=A.dtype, device=A.device)

    A_inv = torch.einsum("iaib->iab", A_inv.reshape(n_in, k, n_in, k))
    A_inv = torch.linalg.inv(A_inv)

    weight_importances = torch.vmap(torch.trace)(A_inv) / 2
    weight_importances = torch.repeat_interleave(weight_importances, k)

    return torch.concat([bias_importance, weight_importances])


def calibrated_importances_activity(A):
    M_inv = torch.linalg.cholesky(A)
    return torch.diag(M_inv.T @ M_inv) / 2


def compute_group_importances(
    input_pruned_layers,
    use_bias,
    eps,
    reduction_fn,
    imp_name,
    local_order=None,
):
    group_importances = []
    for layer in input_pruned_layers:
        n_in, k, uses_bias, alive, local_order_, W, A, G, mu = utils.prepare_vgg_layer(
            layer, local_order, use_bias, eps
        )

        importances_ = compute_layer_importances(
            W, A, G, mu, n_in, k, uses_bias, local_order_, imp_name
        )

        importances = torch.zeros(
            len(alive), dtype=importances_.dtype, device=alive.device
        )
        importances[alive] = importances_[int(uses_bias) :]

        importances = importances.reshape(len(importances) // k, k).sum(1)
        group_importances.append(importances)

    return reduction_fn(group_importances)


def compute_importance_scores(
    DG, ignored_layers, imp_name, reduction_fn, use_bias, eps, normalize=False
):

    local_scores, calibrated_scores = [], []

    for _, input_pruned_layers in pruning.pruning_groups(DG, ignored_layers):
        local = compute_group_importances(
            input_pruned_layers, use_bias, eps, reduction_fn, imp_name
        )
        order = torch.argsort(local, descending=True)

        calibrated = compute_group_importances(
            input_pruned_layers,
            use_bias,
            eps,
            reduction_fn,
            imp_name,
            order,
        )

        if normalize:
            undo_order = torch.argsort(order)
            calibrated = utils.rev_cumsum(calibrated[order])[undo_order]
            calibrated /= calibrated.max()

        local_scores.append(local), calibrated_scores.append(calibrated)

    return local_scores, calibrated_scores


def compute_layer_importances(
    W, A, G, mu, n_in, k, uses_bias, local_order, importance_measure_name
):
    if local_order is None:
        if importance_measure_name == "act":
            return local_importances_activity(A, n_in, k, uses_bias)
        elif importance_measure_name == "tso":
            return local_importances_taylor_second_order(W, A, G, n_in, k, uses_bias)
        elif importance_measure_name == "pfa":
            return local_importances_pfa_en(A, mu, n_in, k)
        elif importance_measure_name == "saw":
            return local_importances_saw(W)
        else:
            raise NotImplementedError()

    i = int(uses_bias)
    local_order_ = torch.concat([torch.arange(i, device=A.device), i + local_order])

    A = A[local_order_][:, local_order_]
    mu = mu[local_order_]
    W = W[:, local_order_]

    if importance_measure_name == "act":
        global_importances = calibrated_importances_activity(A)
    elif importance_measure_name == "tso":
        global_importances = calibrated_importances_taylor_second_order(W, A, G)
    elif importance_measure_name == "pfa":
        global_importances = global_importances_pfa_en(A, mu, n_in, k, local_order_)
    elif importance_measure_name == "saw":
        global_importances = global_importances_saw(W, local_order_, n_in, k)
    else:
        raise NotImplementedError()

    undo_order = torch.argsort(local_order)
    global_importances[i:] = global_importances[i:][undo_order]

    return torch.concat([global_importances[:i], global_importances[i:]])


def local_importances_saw(W):
    return torch.einsum("ij->j", W.abs())


def global_importances_saw(W, local_order, n_in, k):
    scores = torch.arange(len(local_order) // k, dtype=W.dtype, device=W.device)
    scores = torch.flip(scores, [0])
    scores /= (len(scores) - 1) * k
    scores = torch.repeat_interleave(scores, k)

    return scores


def local_importances_pfa_en(A, mu, n_in, k):
    k_sqrt = k ** (1 / 2)
    A = A.reshape(n_in, k, n_in, k)
    A = torch.einsum("iajb->ij", A) / k_sqrt

    stds = A.diagonal().sqrt()
    A = (A / (stds[:, None] * stds[None, :])).abs()
    A.fill_diagonal_(0.0)

    scores = A.sum(dim=1)
    remains = torch.ones(len(A), dtype=torch.bool, device=A.device)
    order = torch.empty(len(A), dtype=A.dtype, device=A.device)

    for i in range(len(A)):
        s = scores.masked_fill(~remains, -1)
        ties = torch.nonzero(torch.isclose(s, s.max())).squeeze(1)

        best_pair = A[ties][:, remains].max(dim=1).values
        j = ties[best_pair.argmax()]

        order[i] = j
        remains[j] = False
        scores -= A[:, j]

    order = torch.repeat_interleave(order / k, k)

    return order


def global_importances_pfa_en(A, mu, n_in, k, local_order):
    k_sqrt = k ** (1 / 2)
    A = A.reshape(n_in, k, n_in, k)
    A = torch.einsum("iajb->ij", A) / k_sqrt
    global_scores = torch.linalg.eigvalsh(A)  # ascending order

    global_scores = torch.flip(global_scores, [0])
    global_scores = torch.repeat_interleave(global_scores / k, k)

    undo_order = torch.argsort(local_order)

    return global_scores[undo_order]


###### reduction functions ######
stack_mean = lambda x: torch.stack(x).mean(dim=0)
first_layer = lambda x: x[0]
last_layer = lambda x: x[-1]
