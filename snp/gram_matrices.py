import torch
from torch import nn

import numpy as np
from tqdm import tqdm

import abc


class StatsCollector(abc.ABC):
    def __init__(self, name: str = None, is_forward_hook: bool = True):
        self._struct = None

        self._hooks = []
        self._accum = {}

        self.name = name if name else "stats"

        if is_forward_hook:
            self.register_hook = lambda mod: mod.register_forward_hook
        else:
            self.register_hook = lambda mod: mod.register_full_backward_hook

    def detach_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    @torch.no_grad()
    def __layer_hook(self, module, args, y):
        x = args[0]
        y = y[0]

        if y is None:
            return

        if self._struct is not None:
            x = torch.reshape(x, self._struct)
            self._struct = None

        ret, batch_size = self._layer_hook(module, x.squeeze(), y.squeeze())

        self._accum[module.name_in_model]["gram"] += ret.cpu().float()
        self._accum[module.name_in_model]["n_samples"] += batch_size

    @abc.abstractmethod
    def _layer_hook(self, module, args, y) -> torch.Tensor:
        """Compute and return a batch-wise estimate of the tracked stats."""

    def _struct_hook(self, module, args, output):
        self._struct = output.shape

    def attach_hooks(self, net, module_before_flatten=None):
        if module_before_flatten is None:
            module_before_flatten = type(None)

        for _, mod in net.named_modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d)):
                handle = self.register_hook(mod)(self.__layer_hook)
                self._hooks.append(handle)
                self._accum[mod.name_in_model] = {"gram": 0.0, "n_samples": 0.0}
            elif isinstance(mod, module_before_flatten):
                handle = self.register_hook(mod)(self._struct_hook)
                self._hooks.append(handle)

    def collect(self):
        return {
            k: {self.name: self._accum[k]["gram"] / self._accum[k]["n_samples"]}
            for k in self._accum.keys()
        }


class InputMeanCollector(StatsCollector):
    @torch.no_grad()
    def _layer_hook(self, module, args, y):
        if args.ndim == 3:
            return args.clone().sum(dim=(0, 1)), args.shape[0] * args.shape[1]
        elif args.ndim == 4:
            return (
                args.clone().sum(dim=(0, 2, 3)),
                args.shape[0]
                * args.shape[2]
                * args.shape[
                    3
                ],  # mean over batches and input HxW - output in shape C_in
            )
        else:
            return args.clone().sum(dim=0), args.shape[0]


class InputPatchMeanCollector(StatsCollector):
    def __init__(self, patch_mean=False, name=None, is_forward_hook=True):
        super().__init__(name, is_forward_hook)
        self.patch_mean = patch_mean

    @torch.no_grad()
    def _layer_hook(self, module, args, y):
        if args.ndim == 3:
            return (
                args.clone().sum(dim=(0, 1)),
                args.shape[0] * args.shape[1],
            )  # transformer
        elif args.ndim == 4:
            # the last layers input must be treated as a 7x7 convolution with stride=1, dilation=1, padding=0
            # we have 25088 input weights -> 7x7x512
            if isinstance(module, nn.Linear):  # input is reshaped due to struct hook
                args = torch.nn.functional.unfold(
                    args, kernel_size=(7, 7), stride=1, dilation=1, padding=0
                )
            else:
                args = torch.nn.functional.unfold(
                    args,
                    kernel_size=module.weight.shape[2:],
                    stride=module.stride,
                    dilation=module.dilation,
                    padding=module.padding,
                )
            return (
                args.clone().sum(dim=(0, 2)),
                (
                    args.shape[0] * args.shape[2] if self.patch_mean else 1
                ),  # only if explicitly asked to demean over num_patches
            )
        else:
            return args.clone().sum(dim=0), args.shape[0]


class InputPatchGramCollector(StatsCollector):
    def __init__(self, means=None, name=None, is_forward_hook=True):
        super().__init__(name, is_forward_hook)

        self.mus = means  # demean if means are provided

    @torch.no_grad()
    def _layer_hook(self, module, x, y):
        x = x.clone()

        if x.ndim == 4:
            if isinstance(module, nn.Linear):
                x = torch.nn.functional.unfold(
                    x, kernel_size=(7, 7), stride=1, dilation=1, padding=0
                )
            else:
                x = torch.nn.functional.unfold(
                    x,
                    kernel_size=module.weight.shape[2:],
                    stride=module.stride,
                    dilation=module.dilation,
                    padding=module.padding,
                )

        if self.mus is not None:
            mu = self.mus[module.name_in_model]
            x = x - mu.reshape(1, -1, *[1] * (x.ndim - 2))

        C = torch.einsum("bi...,bj...->ij", x, x)

        # additive error across patches -> do not normalize over patches!
        return C, len(x)


class InputGramCollector(StatsCollector):
    @torch.no_grad()
    def _layer_hook(self, module, x, y):
        x = x.clone()

        batch_size = len(x) if x.ndim <= 2 else len(x) * np.prod(x.shape[2:])
        C = torch.einsum("bi...,bj...->ij", x, x)

        return C, batch_size


class OutputGradGramCollector(StatsCollector):
    def __init__(self, name: str = None):
        super().__init__(name, is_forward_hook=False)

    @torch.no_grad()
    def _layer_hook(self, module, input_grad, output_grad):
        x = output_grad.clone()

        batch_size = len(x) if x.ndim <= 2 else len(x) * np.prod(x.shape[2:])
        C = torch.einsum("bi...,bj...->ij", x, x)

        return C, batch_size


def run_collector(
    forward_function, net, collectors, modules_before_flatten, dataloader, device
):
    for c, m in zip(collectors, modules_before_flatten):
        c.attach_hooks(net, m)

    net.eval()
    net.to(device)
    for x, y in tqdm(dataloader):
        x = x.to(device)
        x = x.requires_grad_(True)
        y = y.to(device)

        out = net(x)

        forward_function(out, y)

    net.zero_grad(set_to_none=True)

    collections = []
    for c in collectors:
        c.detach_hooks()
        collections.append(c.collect())

    return collections


def run_llm_collector(
    forward_function, net, collectors, modules_before_flatten, dataloader, device
):
    input_ids_full, seq_len, stride, max_len = dataloader
    for c, m in zip(collectors, modules_before_flatten):
        c.attach_hooks(net, m)

    net.eval()
    net.to(device)

    for begin in tqdm(range(0, seq_len, stride)):
        end = min(begin + max_len, seq_len)
        input_ids = input_ids_full[:, begin:end]
        out = net(input_ids).logits
        forward_function(out, None)
        if end == seq_len:
            break

    net.zero_grad(set_to_none=True)

    collections = []
    for c in collectors:
        c.detach_hooks()
        collections.append(c.collect())

    return collections


def collect_cce_fisher_function(outputs, y):
    softmax_outputs = nn.functional.softmax(outputs, dim=1)

    sampled_y = torch.multinomial(softmax_outputs, 1).squeeze(dim=1)
    sample_loss = nn.functional.cross_entropy(outputs, sampled_y, reduction="sum")

    with torch.no_grad():
        sample_loss.backward()


def collect_sse_fisher_function(outputs, y, n_samples=1):
    for s in range(n_samples):
        sampled_y = outputs.detach() + torch.randn_like(outputs)
        loss = 0.5 * ((outputs - sampled_y) ** 2).sum()
        loss.backward(retain_graph=s < n_samples - 1)


def collect_inputs_function(outputs, y):
    pass


def compute_gram_matrices(model, dataloader, llm=False, means=None, device="cuda"):
    for mod in model.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = False

    collectors = [
        InputMeanCollector(name="input_mean"),
        InputPatchMeanCollector(name="input_patch_mean"),
        InputPatchGramCollector(means=means, name="input_patch"),
        InputGramCollector(name="input"),
        OutputGradGramCollector(name="output_grad"),
    ]
    before_flatten = [
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool2d,
        None,
    ]

    run_function = run_llm_collector if llm else run_collector
    collections = run_function(
        collect_sse_fisher_function,
        model,
        collectors,
        before_flatten,
        dataloader,
        device,
    )

    return _merge_collections(collections)


@torch.no_grad()
def compute_means(model, dataloader, device="cuda"):
    for mod in model.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = False

    collectors = [
        InputPatchMeanCollector(patch_mean=True, name="input_patch_mean")
    ]  # patch mean only for covariance matrices
    before_flatten = [nn.AdaptiveAvgPool2d]

    collections = run_collector(
        collect_inputs_function,
        model,
        collectors,
        before_flatten,
        dataloader,
        device,
    )

    collections = _merge_collections(collections)
    return {key: collections[key]["input_patch_mean"] for key in collections.keys()}


def _merge_collections(collections):
    keys = set().union(*(c.keys() for c in collections))

    collections = {
        k: {key: value for c in collections for key, value in c.get(k, {}).items()}
        for k in keys
    }

    return collections
