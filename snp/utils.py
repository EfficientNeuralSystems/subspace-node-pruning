import torch
from torch import nn
import torchvision

import numpy as np
from tqdm import tqdm

from pathlib import Path


def load_model(model_name):
    if model_name == "vgg16":
        return torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        ).eval()
    if model_name == "resnet50":
        return torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        ).eval()
    else:
        raise NotImplementedError()


def assign_names(model):
    for name, mod in model.named_modules():
        mod.name_in_model = name


def set_gram_matrices(model, gram_matrices):
    layers = filter(
        lambda mod: isinstance(mod, (nn.Linear, nn.Conv2d)), model.modules()
    )

    for l in layers:
        setattr(
            l, "input_patch_gram_matrix", gram_matrices[l.name_in_model]["input_patch"]
        )
        setattr(l, "input_gram_matrix", gram_matrices[l.name_in_model]["input"])
        setattr(
            l,
            "input_patch_mean_vector",
            gram_matrices[l.name_in_model]["input_patch_mean"],
        )
        setattr(l, "input_mean_vector", gram_matrices[l.name_in_model]["input_mean"])
        setattr(
            l, "output_grad_gram_matrix", gram_matrices[l.name_in_model]["output_grad"]
        )


@torch.no_grad()
def measure_perf(net, loss_fn, dataloader, device):
    net.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)

        out = net(x)
        total_loss += loss_fn(out, y).item() * len(x)
        total_correct += (out.argmax(dim=1) == y).sum().item()
        total_samples += len(x)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def _make_subset(dataset, n_samples, seed=0, stratified=False):
    if n_samples is None or n_samples == -1 or n_samples >= len(dataset):
        return dataset
    if n_samples < 1:
        raise ValueError(
            f"n_samples must be -1 (all) or a positive int, got {n_samples}"
        )

    g = torch.Generator().manual_seed(seed)

    if stratified:
        # Roughly equal number of samples per class
        targets = torch.as_tensor(dataset.targets)
        classes = targets.unique()
        per_class, extra = divmod(n_samples, len(classes))
        chosen = []
        for i, c in enumerate(classes):
            cls_idx = (targets == c).nonzero().squeeze(1)
            k = per_class + (1 if i < extra else 0)
            chosen.append(cls_idx[torch.randperm(len(cls_idx), generator=g)[:k]])
        indices = torch.cat(chosen)
    else:
        indices = torch.randperm(len(dataset), generator=g)[:n_samples]

    return torch.utils.data.Subset(dataset, indices.tolist())


def load_data(
    data_path,
    batch_size,
    v2=False,
    n_train_samples=-1,
    n_test_samples=-1,
    stratified=False,
    seed=0,
):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if v2:
        test_transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    else:
        test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ]
        )

    train_dataset = torchvision.datasets.ImageFolder(
        root=Path(data_path).joinpath("train"), transform=test_transform
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=Path(data_path).joinpath("val2"), transform=test_transform
    )

    train_dataset = _make_subset(train_dataset, n_train_samples, seed, stratified)
    test_dataset = _make_subset(test_dataset, n_test_samples, seed, stratified)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    return train_dataloader, test_dataloader


def set_num_patches(model, example_input):
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    handles = []

    def make_hook(layer):
        def hook(module, input, output):
            layer.input_pixels = np.prod(input[0].shape[2:])
            layer.num_patches = np.prod(output.shape[2:])

        return hook

    for module in model.modules():
        if isinstance(module, conv_types):
            handles.append(module.register_forward_hook(make_hook(module)))

    model.eval()
    with torch.no_grad():
        model(example_input)

    for h in handles:
        h.remove()


def prepare_params(*params):
    return [p.double().cpu() if p is not None else None for p in params]


def prepare_vgg_layer(layer, local_order, use_bias, eps=1e-12):
    W, b, A, mu, G = prepare_params(
        layer.weight,
        layer.bias,
        layer.input_patch_gram_matrix,
        layer.input_patch_mean_vector,
        layer.output_grad_gram_matrix,
    )

    n_in, k = W.shape[1], np.prod(W.shape[2:]) if W.ndim > 2 else 1
    if n_in == 25088:
        n_in, k = 512, 49

    W = W.reshape(len(W), n_in * k)

    if local_order is not None:
        local_order = block_expand(local_order, k)

    alive, A, W, mu, local_order = remove_dead_units(
        A,
        W,
        mu,
        local_order,
    )
    n_in -= torch.sum(~alive) // k

    uses_bias = b is not None and use_bias
    if uses_bias:
        num_patches = layer.num_patches if hasattr(layer, "num_patches") else 1
        A, W, mu = add_bias(b, W, A, mu, num_patches)

    A = A + A.diagonal().mean() * eps * torch.eye(
        len(A), dtype=A.dtype, device=A.device
    )

    return n_in, k, uses_bias, alive, local_order, W, A, G, mu


def block_expand(idx, k):
    grid = idx[:, None] * k + torch.arange(k, device=idx.device)
    grid = grid.reshape(-1)
    return grid


def rev_cumsum(l):
    return torch.flip(torch.cumsum(torch.flip(l, [0]), 0), [0])


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


def remove_dead_units(A, W, mu, order=None):
    alive = (
        A.diagonal() != 0
    )  # Note: dead units are practically impossible in conv nets (except 1x1 convs),
    #       therefore we refrain from implementing special treatment beyond 1x1 convs
    return (
        alive,
        A[alive][:, alive],
        W[:, alive],
        mu[alive],
        (
            (torch.cumsum(alive, 0) - 1)[order[alive[order]]]
            if order is not None
            else None
        ),
    )


def add_bias(b, W, A, mean, num_patches=1):
    n_in, n_out = W.shape[1], len(b)

    A_ = torch.zeros(n_in + 1, n_in + 1, dtype=A.dtype, device=A.device)
    A_[1:, 1:] = A
    A_[1:, 0] = mean
    A_[0, 1:] = mean
    A_[0, 0] = num_patches

    W_ = torch.zeros(n_out, n_in + 1, dtype=W.dtype, device=W.device)
    W_[:, 1:] = W
    W_[:, 0] = b

    mean_ = torch.zeros(n_in + 1, dtype=mean.dtype, device=mean.device)
    mean_[1:] = mean
    mean_[0] = num_patches

    return A_, W_, mean_
