# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ported from ultralytics_sonnn nn/modules/moe/modules.py (model zoo; see ultralytics/nn/zoo/__init__.py)."""

from ultralytics.nn.zoo.moe.experts import EfficientExpertGroup
from ultralytics.nn.zoo.moe.experts import FasterExpert
from ultralytics.nn.zoo.moe.experts import GBCExpert
from ultralytics.nn.zoo.moe.experts import GhostExpert
from ultralytics.nn.zoo.moe.experts import InvertedResidualExpert
from ultralytics.nn.zoo.moe.experts import ShuffleExpert
from ultralytics.nn.zoo.moe.experts import SimpleExpert
from ultralytics.nn.zoo.moe.experts import SpatialExpert
from ultralytics.nn.zoo.moe.experts import UniversalInvertedResidualExpert
from ultralytics.nn.zoo.moe.loss import MoELoss
from ultralytics.nn.zoo.moe.routers import AdaptiveRoutingLayer
from ultralytics.nn.zoo.moe.routers import DynamicRoutingLayer
from ultralytics.nn.zoo.moe.routers import EfficientSpatialRouter
from ultralytics.nn.zoo.moe.routers import LocalRoutingLayer
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref


MOE_LOSS_REGISTRY = weakref.WeakKeyDictionary()


def _robust_deepcopy(obj, memo):
    """
    Robust deepcopy helper that sanitizes the object's __dict__ to remove
    any non-leaf tensors (which cause RuntimeError in deepcopy) before copying.
    """
    cls = obj.__class__
    new_obj = cls.__new__(cls)
    memo[id(obj)] = new_obj

    for k, v in obj.__dict__.items():
        # Check for non-leaf tensor (has grad_fn)
        if isinstance(v, torch.Tensor) and v.grad_fn is not None:
            # Replace with a safe scalar zero (detached)
            setattr(new_obj, k, torch.tensor(0.0))
        else:
            try:
                setattr(new_obj, k, copy.deepcopy(v, memo))
            except RuntimeError as e:
                # Fallback: if deepcopy fails on a specific attribute, try to skip or reset it
                if "Only Tensors created explicitly" in str(e):
                    print(
                        f"WARNING: Skipped deepcopy for attribute '{k}' in {cls.__name__} due to non-leaf tensor error."
                    )
                    setattr(new_obj, k, torch.tensor(0.0))
                else:
                    raise e
            except Exception:
                # Best effort copy for other errors (e.g. pickling issues)
                # If it fails, we assume it's transient state and ignore it or shallow copy
                setattr(new_obj, k, v)

    return new_obj


class ES_MOE(nn.Module):
    """General MoE block with a routing network and multiple expert branches."""

    def __init__(
        self,
        in_channels,
        out_channels=None,
        num_experts=3,
        reduction=8,
        top_k=None,
        use_sparse_inference=True,
        dynamic_threshold=0.4,
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels (defaults to in_channels)
            num_experts: Number of expert branches
            reduction: Channel reduction ratio for the routing network
            top_k: Number of active experts; None means use all experts
            use_sparse_inference: Enable sparse Top-K expert computation during inference
            dynamic_threshold: Threshold for pruning low-confidence experts during inference
        """
        super(ES_MOE, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts) if top_k is not None else num_experts
        self.use_top_k = top_k is not None
        self.use_sparse_inference = use_sparse_inference
        self.dynamic_threshold = dynamic_threshold

        # Dynamic routing (Top-K supported)
        self.routing = DynamicRoutingLayer(in_channels, num_experts, reduction, top_k)

        # Expert group (original design)
        default_kernel_sizes = [3, 5, 7]
        if num_experts <= len(default_kernel_sizes):
            ks = default_kernel_sizes[:num_experts]
        else:
            ks = [3 + 2 * i for i in range(num_experts)]
        self.experts = nn.ModuleList([EfficientExpertGroup(in_channels, out_channels, kernel_size=k) for k in ks])

        # Output normalization (original design)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        # Load-balancing loss (original design)
        self.register_buffer("load_balancing_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer("expert_usage_counts", torch.zeros(num_experts), persistent=False)

    def forward(self, x):
        if not hasattr(self, "use_top_k"):
            self.use_top_k = False
        if not hasattr(self, "use_sparse_inference"):
            self.use_sparse_inference = True
        if not hasattr(self, "num_experts"):
            self.num_experts = len(self.experts) if hasattr(self, "experts") else 1
        if not hasattr(self, "top_k"):
            self.top_k = self.num_experts
        # Get routing weights
        routing_weights = self.routing(x)

        # Compute load-balancing loss
        self._compute_load_balancing_loss(routing_weights)

        # Different forward strategies for train/infer
        if self.training or not self.use_top_k or not self.use_sparse_inference:
            # Train mode or no Top-K or no sparse inference: dense compute
            final_output = self._dense_forward(x, routing_weights)
        else:
            # Infer mode + Top-K + sparse inference: compute Top-K experts only
            final_output = self._sparse_forward(x, routing_weights.detach())

        if not hasattr(self, "norm"):
            self.norm = nn.Sequential(
                nn.BatchNorm2d(final_output.shape[1]),
                nn.SiLU(inplace=True),
            )
        final_output = self.norm(final_output)

        return final_output

    @property
    def aux_loss(self):
        """Retrieve the auxiliary loss from the registry."""
        return MOE_LOSS_REGISTRY.get(self, torch.tensor(0.0))

    def _dense_forward(self, x, routing_weights):
        """Dense forward: compute all experts (used during training)."""
        final_output = 0
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            weight = routing_weights[:, i : i + 1, :, :]
            final_output = final_output + expert_out * weight
        return final_output

    def _sparse_forward(self, x, routing_weights):
        """Sparse forward: compute only Top-K experts (used during inference)."""
        B, E, H, W = routing_weights.shape

        # Compute per-expert importance
        routing_weights_flat = routing_weights.view(B, E, -1)
        expert_importance = routing_weights_flat.mean(dim=2)

        # Find Top-K experts
        topk_values, topk_indices = torch.topk(expert_importance, self.top_k, dim=1)

        # Initialize output
        final_output = torch.zeros_like(x)

        # Iterate over experts (vectorized over batch)
        for expert_idx in range(self.num_experts):
            # Find batch samples that selected this expert
            mask = topk_indices == expert_idx
            if not mask.any():
                continue

            batch_indices, k_ranks = torch.where(mask)

            # === Dynamic Pruning ===
            if hasattr(self, "dynamic_threshold") and self.dynamic_threshold > 0:
                current_weights = routing_weights[batch_indices, expert_idx : expert_idx + 1, :, :]
                # Keep if (rank == 0) OR (weight >= threshold)
                weight_means = current_weights.mean(dim=(1, 2, 3))
                keep_mask = (k_ranks == 0) | (weight_means >= self.dynamic_threshold)

                batch_indices = batch_indices[keep_mask]
                if batch_indices.numel() == 0:
                    continue
            # =======================

            # Compute expert output for selected samples
            expert_out = self.experts[expert_idx](x[batch_indices])
            weight = routing_weights[batch_indices, expert_idx : expert_idx + 1, :, :]

            # Accumulate
            final_output.index_add_(0, batch_indices, expert_out * weight)

        return final_output

    def _compute_load_balancing_loss(self, routing_weights, eps=1e-6):
        """Compute load-balancing loss (original logic)."""
        expert_usage = routing_weights.mean(dim=(0, 2, 3))
        ideal_usage = 1.0 / self.num_experts
        load_balance_loss = F.mse_loss(expert_usage, torch.full_like(expert_usage, ideal_usage))

        # Guard against NaN loss
        if torch.isnan(load_balance_loss):
            load_balance_loss = torch.tensor(0.0, device=load_balance_loss.device, requires_grad=True)

        if not hasattr(self, "load_balancing_loss"):
            self.register_buffer("load_balancing_loss", torch.tensor(0.0), persistent=False)
        if not hasattr(self, "expert_usage_counts"):
            self.register_buffer("expert_usage_counts", torch.zeros_like(expert_usage), persistent=False)
        if self.load_balancing_loss.shape == torch.Size([]):
            self.load_balancing_loss = self.load_balancing_loss.to(load_balance_loss.device).reshape(())
        self.load_balancing_loss.copy_(load_balance_loss.detach())
        self.expert_usage_counts.copy_(expert_usage.detach())

        # Store in registry
        MOE_LOSS_REGISTRY[self] = load_balance_loss

        return load_balance_loss

    def get_load_balancing_loss(self):
        """Get load-balancing loss."""
        return self.load_balancing_loss

    def get_expert_usage_stats(self):
        """Get expert usage statistics."""
        if self.expert_usage_counts.numel() > 0:
            stats = {
                "expert_usage": self.expert_usage_counts.cpu().tolist(),
                "usage_variance": self.expert_usage_counts.var().item(),
                "max_usage": self.expert_usage_counts.max().item(),
                "min_usage": self.expert_usage_counts.min().item(),
            }
            if self.use_top_k:
                stats["active_experts"] = f"{self.top_k}/{self.num_experts}"
                stats["theoretical_speedup"] = f"{self.num_experts / self.top_k:.2f}x"
            return stats
        return None

    def set_top_k(self, top_k):
        """Dynamically adjust Top-K value."""
        if top_k is not None:
            self.top_k = min(top_k, self.num_experts)
            self.routing.top_k = self.top_k
            self.use_top_k = True
            self.routing.use_top_k = True
        else:
            self.top_k = self.num_experts
            self.use_top_k = False
            self.routing.use_top_k = False

    def enable_sparse_inference(self, enable=True):
        """Enable/disable sparse inference."""
        self.use_sparse_inference = enable

    def __deepcopy__(self, memo):
        return _robust_deepcopy(self, memo)


class OptimizedMOEImproved(nn.Module):
    """Improved MoE with pluggable routers/experts and a shared expert for stability."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_type: str = "simple",  # ['simple', 'ghost', 'inverted']
        router_type: str = "efficient",  # ['simple', 'ghost', 'inverted', 'spatial']
        noise_std: float = 1.0,
        balance_loss_coeff: float = 0.01,
        router_z_loss_coeff: float = 1e-3,
        expert_expand_ratio: float = 2.0,
        progressive_sparsity: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_coeff = balance_loss_coeff
        self.router_z_loss_coeff = router_z_loss_coeff
        self.progressive_sparsity = progressive_sparsity

        # Progressive Sparsity
        self.register_buffer("training_step", torch.tensor(0))
        self.register_buffer("current_top_k", torch.tensor(num_experts))
        self.warmup_steps = 5000

        # 1) Instantiate Router
        if router_type == "local":
            self.routing = LocalRoutingLayer(in_channels, num_experts, top_k=top_k, noise_std=noise_std)
        elif router_type == "adaptive":
            self.routing = AdaptiveRoutingLayer(in_channels, num_experts, top_k=top_k, noise_std=noise_std)
        else:
            self.routing = EfficientSpatialRouter(in_channels, num_experts, top_k=top_k, noise_std=noise_std)

        # 2) Instantiate Experts
        self.experts = nn.ModuleList()
        kwargs = {}
        if expert_type == "ghost":
            expert_cls = GhostExpert
            kwargs["ratio"] = int(expert_expand_ratio)
        elif expert_type == "inverted":
            expert_cls = InvertedResidualExpert
            kwargs["ratio"] = int(expert_expand_ratio)
        elif expert_type == "spatial":
            expert_cls = SpatialExpert
            kwargs["expand_ratio"] = expert_expand_ratio
        elif expert_type == "universalinverted":
            expert_cls = UniversalInvertedResidualExpert
            kwargs["expand_ratio"] = expert_expand_ratio
        elif expert_type == "shuffle":
            expert_cls = ShuffleExpert
            kwargs["expand_ratio"] = expert_expand_ratio
        elif expert_type == "faster":
            expert_cls = FasterExpert
            kwargs["expand_ratio"] = expert_expand_ratio
        elif expert_type == "gbc":
            expert_cls = GBCExpert
            kwargs["expand_ratio"] = expert_expand_ratio
        else:
            expert_cls = SimpleExpert
            kwargs["expand_ratio"] = expert_expand_ratio

        for _ in range(num_experts):
            self.experts.append(expert_cls(in_channels, out_channels, **kwargs))

        # 3) Shared expert (Always active)
        self.shared_expert = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True)
        )

        self._init_weights()
        self.moe_loss_fn = MoELoss(
            balance_loss_coeff=balance_loss_coeff,
            z_loss_coeff=router_z_loss_coeff,
            num_experts=num_experts,
            top_k=top_k,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Robust router init: find the last Conv layer to initialize
        # Keep initial expert probabilities nearly uniform
        for m in self.routing.router.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv:
            nn.init.normal_(last_conv.weight, mean=0, std=0.01)
            if last_conv.bias is not None:
                nn.init.constant_(last_conv.bias, 0)

    def _update_sparsity(self):
        """Progressive Sparsity Scheduling"""
        if self.training_step < self.warmup_steps:
            progress = self.training_step.float() / self.warmup_steps
            current_k = self.num_experts - progress * (self.num_experts - self.top_k)
            self.current_top_k.fill_(max(self.top_k, int(current_k)))
        else:
            self.current_top_k.fill_(self.top_k)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.training and self.progressive_sparsity:
            self._update_sparsity()
            self.training_step += 1

        # Use current_top_k for routing
        adaptive_top_k = int(self.current_top_k.item()) if self.training and self.progressive_sparsity else self.top_k

        # Temporarily modify routing.top_k
        original_top_k = self.routing.top_k
        self.routing.top_k = adaptive_top_k

        # 1) Routing (standardized interface)
        # loss_dict contains training loss inputs; empty during inference
        routing_weights, routing_indices, loss_dict = self.routing(x)

        # Restore routing.top_k
        self.routing.top_k = original_top_k

        # 2) Shared expert compute
        shared_out = self.shared_expert(x)

        # 3) Sparse expert compute
        # Initialize outputs with zeros
        expert_output = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)

        indices_flat = routing_indices.view(B, adaptive_top_k)
        weights_flat = routing_weights.view(B, adaptive_top_k)

        for i in range(self.num_experts):
            # Find all samples assigned to expert i
            mask = indices_flat == i
            if mask.any():
                batch_idx, k_idx = torch.where(mask)

                # Select input and compute
                inp = x[batch_idx]
                out = self.experts[i](inp)

                # Select weights and broadcast
                w = weights_flat[batch_idx, k_idx].view(-1, 1, 1, 1)

                # Accumulate results
                expert_output.index_add_(0, batch_idx, out.to(expert_output.dtype) * w.to(expert_output.dtype))

        final_output = shared_out + expert_output

        # Add residual connection if dimensions match
        if self.in_channels == self.out_channels:
            final_output = final_output + x

        # 4) Compute and return Loss during training
        if self.training and loss_dict:
            aux_loss = self.moe_loss_fn(
                loss_dict["router_probs"], loss_dict["router_logits"], loss_dict["topk_indices"]
            )
            MOE_LOSS_REGISTRY[self] = aux_loss
        else:
            pass

        return final_output


ModularRouterExpertMoE = OptimizedMOEImproved
