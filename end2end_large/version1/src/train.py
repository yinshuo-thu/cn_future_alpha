from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import time

from .data import RollingSplit, WindowDataset, collect_aux_targets, collect_targets
from .metrics import compute_ic, compute_rank_ic
from .models import EndToEndTransformerBaseline, combo_loss, ic_loss


def _feature_base_name(col: str) -> str:
    return col[3:] if col.startswith("rz_") else col


def _resolve_stream_groups(
    feature_cols: list[str] | None,
    n_features: int,
    groups_cfg: dict[str, list[str]] | None,
) -> list[list[int]] | None:
    if not feature_cols:
        return None
    bases = [_feature_base_name(col) for col in feature_cols]
    groups: list[list[int]] = []
    used: set[int] = set()
    if groups_cfg:
        for patterns in groups_cfg.values():
            idx: list[int] = []
            for pattern in patterns:
                pat = str(pattern)
                for i, (col, base) in enumerate(zip(feature_cols, bases)):
                    if i in used:
                        continue
                    if col == pat or base == pat or col.endswith(pat) or base.endswith(pat):
                        idx.append(i)
                        used.add(i)
            if idx:
                groups.append(sorted(idx))
    else:
        price_names = {"open", "high", "low", "close", "close_pos", "oc_ret"}
        vol_names = {"log_ret_1", "range_pct"}
        flow_tokens = ("volume", "amount", "oi")
        price = [i for i, base in enumerate(bases) if base in price_names]
        vol = [i for i, base in enumerate(bases) if base in vol_names]
        flow = [i for i, base in enumerate(bases) if any(tok in base for tok in flow_tokens)]
        for group in (price, flow, vol):
            idx = [i for i in group if i not in used]
            used.update(idx)
            if idx:
                groups.append(idx)
    leftovers = [i for i in range(n_features) if i not in used]
    if leftovers:
        groups.append(leftovers)
    return groups or None


def make_model(
    config: dict[str, Any],
    n_features: int,
    n_symbols: int,
    feature_cols: list[str] | None = None,
) -> EndToEndTransformerBaseline:
    model_cfg = config["model"]
    multi_patch_scales = model_cfg.get("multi_patch_scales")
    if multi_patch_scales:
        multi_patch_scales = [tuple(int(v) for v in scale) for scale in multi_patch_scales]
    stream_groups = None
    if model_cfg.get("input_block", "lowrank") == "stream":
        stream_groups = _resolve_stream_groups(
            feature_cols or config.get("data", {}).get("feature_cols"),
            n_features,
            model_cfg.get("stream_groups"),
        )
    return EndToEndTransformerBaseline(
        n_features=n_features,
        n_symbols=n_symbols,
        seq_len=config["data"]["seq_len"],
        interaction_dim=model_cfg.get("interaction_dim", 64),
        interaction_rank=model_cfg.get("interaction_rank", 8),
        d_model=model_cfg.get("d_model", 128),
        n_heads=model_cfg.get("n_heads", 4),
        n_layers=model_cfg.get("n_layers", 4),
        ffn_dim=model_cfg.get("ffn_dim", model_cfg.get("d_model", 128) * 2),
        patch_len=model_cfg.get("patch_len", 12),
        stride=model_cfg.get("stride", 6),
        dropout=model_cfg.get("dropout", 0.1),
        attention_dropout=model_cfg.get("attention_dropout", 0.1),
        pooling=model_cfg.get("pooling", "attention"),
        patch_encoder=model_cfg.get("patch_encoder", "mlp"),
        use_cls_token=model_cfg.get("use_cls_token", False),
        input_block=model_cfg.get("input_block", "lowrank"),
        feature_token_heads=model_cfg.get("feature_token_heads", 4),
        feature_token_layers=model_cfg.get("feature_token_layers", 1),
        multi_patch_scales=multi_patch_scales,
        aux_output_dim=int(model_cfg.get("aux_output_dim", len(config.get("train", {}).get("aux_targets") or []))),
        stream_groups=stream_groups,
        stream_encoder=model_cfg.get("stream_encoder", "lowrank"),
        feature_gate=bool(model_cfg.get("feature_gate", False)),
        feature_gate_dropout=float(model_cfg.get("feature_gate_dropout", 0.0) or 0.0),
        latent_slots=int(model_cfg.get("latent_slots", 0) or 0),
        latent_mode=model_cfg.get("latent_mode", "replace"),
        moe_n_experts=int(model_cfg.get("moe_n_experts", 1) or 1),
        moe_hidden_dim=int(model_cfg.get("moe_hidden_dim", max(16, model_cfg.get("d_model", 128) // 2))),
        moe_temperature=float(model_cfg.get("moe_temperature", 1.0) or 1.0),
        moe_expert_hidden_dim=(
            int(model_cfg["moe_expert_hidden_dim"]) if model_cfg.get("moe_expert_hidden_dim") is not None else None
        ),
        moe_expert_mid_dim=(
            int(model_cfg["moe_expert_mid_dim"]) if model_cfg.get("moe_expert_mid_dim") is not None else None
        ),
        moe_base_n_experts=int(model_cfg.get("moe_base_n_experts", 0) or 0),
        moe_base_hidden_dim=(
            int(model_cfg["moe_base_hidden_dim"]) if model_cfg.get("moe_base_hidden_dim") is not None else None
        ),
        moe_residual_scale_init=float(model_cfg.get("moe_residual_scale_init", 1.0) or 1.0),
        feature_cols=feature_cols or config.get("data", {}).get("feature_cols"),
        factor_operator_bank=bool(model_cfg.get("factor_operator_bank", False)),
        factor_operator_windows=[int(w) for w in model_cfg.get("factor_operator_windows", [])]
        if model_cfg.get("factor_operator_windows")
        else None,
        factor_operator_dropout=float(model_cfg.get("factor_operator_dropout", 0.0) or 0.0),
        factor_operator_clip=float(model_cfg.get("factor_operator_clip", 12.0) or 12.0),
        factor_operator_projection_dim=int(model_cfg.get("factor_operator_projection_dim", 0) or 0),
        factor_operator_gate=model_cfg.get("factor_operator_gate", "none"),
        factor_operator_topk=int(model_cfg.get("factor_operator_topk", 0) or 0),
        factor_operator_topk_mode=model_cfg.get("factor_operator_topk_mode", "hard"),
        factor_operator_soft_temperature=float(model_cfg.get("factor_operator_soft_temperature", 1.0) or 1.0),
        factor_operator_gate_hidden=int(model_cfg.get("factor_operator_gate_hidden", 64) or 64),
        factor_operator_extra_ops=bool(model_cfg.get("factor_operator_extra_ops", False)),
        ffn_type=model_cfg.get("ffn_type", "glu"),
        layer_agg_mode=model_cfg.get("layer_agg_mode", "learned"),
        decomposed_head=bool(model_cfg.get("decomposed_head", False)),
        additive_decomposed_head=bool(model_cfg.get("additive_decomposed_head", False)),
        regime_conditioning=bool(model_cfg.get("regime_conditioning", False)),
        regime_hidden_dim=int(model_cfg.get("regime_hidden_dim", 128) or 128),
        regime_scale_init=float(model_cfg.get("regime_scale_init", 0.1) or 0.1),
        regime_target=model_cfg.get("regime_target", "both"),
        time_bucket_minutes=int(model_cfg.get("time_bucket_minutes", 0) or 0),
        minute_encoding=model_cfg.get("minute_encoding", "embedding"),
        minute_harmonics=int(model_cfg.get("minute_harmonics", 4) or 4),
        use_month_embedding=bool(model_cfg.get("use_month_embedding", True)),
        symbol_embedding_dim=(
            int(model_cfg["symbol_embedding_dim"]) if model_cfg.get("symbol_embedding_dim") is not None else None
        ),
        decomp_market_scale_init=float(model_cfg.get("decomp_market_scale_init", 1.0) or 1.0),
        decomp_residual_scale_init=float(model_cfg.get("decomp_residual_scale_init", 1.0) or 1.0),
        decomp_component_gate=bool(model_cfg.get("decomp_component_gate", False)),
        pool_fusion_modes=[str(v) for v in model_cfg.get("pool_fusion_modes", [])]
        if model_cfg.get("pool_fusion_modes")
        else None,
        pool_fusion_hidden_dim=(
            int(model_cfg["pool_fusion_hidden_dim"]) if model_cfg.get("pool_fusion_hidden_dim") is not None else None
        ),
        head_type=model_cfg.get("head_type", "default"),
        detach_aux_head_input=bool(model_cfg.get("detach_aux_head_input", False)),
        raw_xsz_scale_init=float(model_cfg.get("raw_xsz_scale_init", 0.25) or 0.25),
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _load_initial_state(model: torch.nn.Module, train_cfg: dict[str, Any], split: RollingSplit | None = None) -> None:
    init_path = train_cfg.get("init_checkpoint")
    if not init_path:
        return
    if split is not None:
        init_path = str(init_path).format(split=split.name, split_name=split.name)
    ckpt = torch.load(str(init_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise ValueError(f"init checkpoint {init_path} does not contain a state dict")
    mapped = dict(state)
    if bool(train_cfg.get("init_moe_head_to_base", True)):
        prefix = "moe_head."
        base_prefix = "moe_head.base."
        for key, value in state.items():
            if key.startswith(prefix) and not key.startswith(base_prefix) and not key.startswith("moe_head.residual."):
                mapped[base_prefix + key[len(prefix) :]] = value
    target = model.state_dict()
    loadable = {
        key: value
        for key, value in mapped.items()
        if key in target and hasattr(value, "shape") and tuple(value.shape) == tuple(target[key].shape)
    }
    incompatible = model.load_state_dict(loadable, strict=False)
    skipped = len(mapped) - len(loadable)
    print(
        f"[train] init_checkpoint={init_path} loaded={len(loadable)} "
        f"skipped={skipped} missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
        flush=True,
    )


def _apply_trainable_patterns(model: torch.nn.Module, train_cfg: dict[str, Any]) -> None:
    trainable_patterns = [str(p) for p in _as_list(train_cfg.get("trainable_patterns")) if str(p)]
    freeze_patterns = [str(p) for p in _as_list(train_cfg.get("freeze_patterns")) if str(p)]
    if trainable_patterns:
        for name, param in model.named_parameters():
            param.requires_grad = any(pattern in name for pattern in trainable_patterns)
    if freeze_patterns:
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in freeze_patterns):
                param.requires_grad = False
    if trainable_patterns or freeze_patterns:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[train] trainable_patterns={trainable_patterns or 'ALL'} "
            f"freeze_patterns={freeze_patterns or 'NONE'} trainable={n_trainable:,}/{n_params:,}",
            flush=True,
        )


def _set_frozen_modules_eval(module: torch.nn.Module) -> None:
    for child in module.children():
        _set_frozen_modules_eval(child)
    params = list(module.parameters(recurse=True))
    if params and not any(param.requires_grad for param in params):
        module.eval()


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(pattern in name for pattern in patterns)


def _make_anchor_state(model: torch.nn.Module, train_cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    weight = float(train_cfg.get("init_anchor_weight", 0.0) or 0.0)
    if weight <= 0:
        return {}
    include = [str(p) for p in _as_list(train_cfg.get("init_anchor_patterns")) if str(p)]
    exclude = [str(p) for p in _as_list(train_cfg.get("init_anchor_exclude_patterns")) if str(p)]
    state: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if include and not _matches_any(name, include):
            continue
        if exclude and _matches_any(name, exclude):
            continue
        state[name] = param.detach().clone()
    print(f"[train] init_anchor params={sum(v.numel() for v in state.values()):,} tensors={len(state)}", flush=True)
    return state


def _anchor_l2_loss(model: torch.nn.Module, anchor_state: dict[str, torch.Tensor]) -> torch.Tensor:
    if not anchor_state:
        return next(model.parameters()).new_tensor(0.0)
    loss = next(model.parameters()).new_tensor(0.0)
    count = 0
    for name, param in model.named_parameters():
        ref = anchor_state.get(name)
        if ref is None:
            continue
        loss = loss + (param - ref).pow(2).sum()
        count += int(param.numel())
    if count <= 0:
        return loss
    return loss / float(count)


def _make_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    base_lr = float(train_cfg.get("lr", 3e-4))
    base_wd = float(train_cfg.get("weight_decay", 0.05))
    specs = [spec for spec in _as_list(train_cfg.get("lr_multipliers")) if isinstance(spec, dict)]
    assigned: set[int] = set()
    groups: list[dict[str, Any]] = []
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    for spec in specs:
        pattern = str(spec.get("pattern", ""))
        if not pattern:
            continue
        params = [param for name, param in named_params if pattern in name and id(param) not in assigned]
        if not params:
            continue
        assigned.update(id(param) for param in params)
        lr_mult = float(spec.get("lr_mult", 1.0))
        wd_mult = float(spec.get("weight_decay_mult", 1.0))
        groups.append(
            {
                "params": params,
                "lr": base_lr * lr_mult,
                "weight_decay": base_wd * wd_mult,
            }
        )
        print(
            f"[train] optimizer_group pattern={pattern} params={sum(p.numel() for p in params):,} "
            f"lr={base_lr * lr_mult:.3e} wd={base_wd * wd_mult:.3e}",
            flush=True,
        )
    default_params = [param for _name, param in named_params if id(param) not in assigned]
    if default_params:
        groups.append({"params": default_params, "lr": base_lr, "weight_decay": base_wd})
        print(
            f"[train] optimizer_group pattern=DEFAULT params={sum(p.numel() for p in default_params):,} "
            f"lr={base_lr:.3e} wd={base_wd:.3e}",
            flush=True,
        )
    if not groups:
        raise ValueError("optimizer received no trainable parameters")
    return torch.optim.AdamW(groups)


def _loss(loss_name: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if loss_name == "mse":
        return torch.nn.functional.mse_loss(pred, target)
    if loss_name == "huber":
        return torch.nn.functional.smooth_l1_loss(pred, target, beta=1.0)
    if loss_name == "ic":
        return ic_loss(pred, target)
    if loss_name == "combo":
        return combo_loss(pred, target)
    raise ValueError(f"unknown loss: {loss_name}")


def _pointwise_loss(loss_name: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    if loss_name == "mse":
        return torch.nn.functional.mse_loss(pred, target, reduction="none")
    if loss_name == "huber":
        return torch.nn.functional.smooth_l1_loss(pred, target, beta=1.0, reduction="none")
    return None


def _recency_loss_weights(
    group_ids: torch.Tensor,
    train_end: Any,
    halflife_days: float,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor | None:
    halflife_days = float(halflife_days)
    if halflife_days <= 0:
        return None
    train_end_ns = pd_timestamp_to_ns(train_end)
    days_old = (float(train_end_ns) - group_ids.to(dtype=torch.float64)).clamp_min(0.0) / (86400.0 * 1e9)
    weights = torch.exp((-np.log(2.0) / halflife_days) * days_old).to(dtype=torch.float32)
    min_weight = float(min_weight)
    max_weight = float(max_weight)
    if min_weight > 0 or max_weight > 0:
        lo = min_weight if min_weight > 0 else None
        hi = max_weight if max_weight > 0 else None
        weights = torch.clamp(weights, min=lo, max=hi)
    mean = weights.mean().clamp_min(1e-6)
    return weights / mean


def _cosine_ic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(pred) & torch.isfinite(target)
    pred = pred[mask]
    target = target[mask]
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    num = (pred * target).mean()
    den = torch.sqrt(pred.pow(2).mean() + 1e-8) * torch.sqrt(target.pow(2).mean() + 1e-8)
    return -num / (den + 1e-8)


def pd_timestamp_to_ns(value: Any) -> int:
    return int(np.datetime64(value, "ns").astype(np.int64))


def _cvar_mse_loss(pred: torch.Tensor, target: torch.Tensor, tail_frac: float) -> torch.Tensor:
    mask = torch.isfinite(pred) & torch.isfinite(target)
    if not mask.any():
        return pred.new_tensor(0.0)
    losses = (pred[mask] - target[mask]).pow(2)
    if losses.numel() == 0:
        return pred.new_tensor(0.0)
    k = max(1, int(np.ceil(float(tail_frac) * int(losses.numel()))))
    k = min(k, int(losses.numel()))
    return torch.topk(losses, k=k, largest=True).values.mean()


def _main_output(out: torch.Tensor) -> torch.Tensor:
    return out[:, 0] if out.ndim == 2 else out


def _aux_output(out: torch.Tensor) -> torch.Tensor | None:
    if out.ndim != 2 or out.shape[1] <= 1:
        return None
    return out[:, 1:]


def _prediction_output_from_model(
    out: torch.Tensor,
    target_mean: float,
    target_std: float,
    destandardize: bool,
    output_mode: str = "main",
    aux_mean: np.ndarray | None = None,
    aux_std: np.ndarray | None = None,
    scale_aux_indices: list[int] | None = None,
    scale_power: float = 1.0,
    scale_min: float = 0.25,
    scale_max: float = 4.0,
) -> torch.Tensor:
    main = _main_output(out).float()
    if destandardize:
        main = main * float(target_std) + float(target_mean)
    mode = str(output_mode or "main").lower()
    if mode in {"", "main", "score", "rank"}:
        return main
    if mode not in {"rank_times_aux_scale", "score_times_aux_scale", "aux_scale", "aux", "aux_value"}:
        raise ValueError(f"unknown prediction output mode: {output_mode}")
    aux = _aux_output(out)
    if aux is None or aux.shape[-1] == 0:
        return main
    if aux_mean is None or aux_std is None:
        return main
    mean = torch.as_tensor(aux_mean, device=aux.device, dtype=aux.dtype)
    std = torch.as_tensor(aux_std, device=aux.device, dtype=aux.dtype)
    width = min(int(aux.shape[-1]), int(mean.numel()), int(std.numel()))
    if width <= 0:
        return main
    if scale_aux_indices:
        idx = torch.as_tensor([i for i in scale_aux_indices if 0 <= int(i) < width], device=aux.device, dtype=torch.long)
        if idx.numel() == 0:
            idx = torch.arange(width, device=aux.device, dtype=torch.long)
    else:
        idx = torch.arange(width, device=aux.device, dtype=torch.long)
    if mode in {"aux", "aux_value"}:
        raw_aux = aux[:, idx] * std[idx].view(1, -1) + mean[idx].view(1, -1)
        raw_aux = torch.nan_to_num(raw_aux, nan=0.0, posinf=0.0, neginf=0.0)
        return raw_aux.mean(dim=-1)
    raw_aux = aux[:, idx] * std[idx].view(1, -1) + mean[idx].view(1, -1)
    denom = mean[idx].abs().clamp_min(1e-8).view(1, -1)
    rel_scale = (raw_aux.clamp_min(0.0) / denom).mean(dim=-1)
    rel_scale = torch.nan_to_num(rel_scale, nan=1.0, posinf=float(scale_max), neginf=float(scale_min))
    rel_scale = rel_scale.clamp(min=float(scale_min), max=float(scale_max)).pow(float(scale_power))
    if mode == "aux_scale":
        return rel_scale
    return main * rel_scale


def _datetime_group_id(arrays, sym_id: int, row: int) -> int:
    return int(np.asarray(arrays[sym_id].datetimes[row], dtype="datetime64[ns]").astype(np.int64))


def _group_positions_by_datetime(arrays, index: list[tuple[int, int]]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for pos, (sym_id, row) in enumerate(index):
        gid = _datetime_group_id(arrays, sym_id, row)
        groups.setdefault(gid, []).append(pos)
    return groups


def _limit_index_by_time_groups(
    arrays,
    index: list[tuple[int, int]],
    max_windows: int,
    min_group_size: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if len(index) <= max_windows:
        return index
    groups = _group_positions_by_datetime(arrays, index)
    keys = [key for key, pos in groups.items() if len(pos) >= min_group_size]
    if not keys:
        return index[:max_windows]
    rng.shuffle(keys)
    selected: list[int] = []
    for key in keys:
        pos = groups[key]
        if selected and len(selected) + len(pos) > max_windows:
            continue
        selected.extend(pos)
        if len(selected) >= max_windows:
            break
    if not selected:
        return index[:max_windows]
    selected.sort()
    return [index[pos] for pos in selected]


def _sample_index_by_recency(
    arrays,
    index: list[tuple[int, int]],
    max_windows: int,
    halflife_days: float,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if len(index) <= max_windows:
        return index
    halflife_days = float(halflife_days)
    if halflife_days <= 0:
        chosen = rng.choice(len(index), size=max_windows, replace=False)
        return [index[int(i)] for i in chosen]
    times = np.fromiter(
        (_datetime_group_id(arrays, sym_id, row) for sym_id, row in index),
        dtype=np.int64,
        count=len(index),
    )
    age_days = (float(times.max()) - times.astype(np.float64)) / (1e9 * 86400.0)
    weights = np.exp(-np.log(2.0) * age_days / halflife_days)
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()
    chosen = rng.choice(len(index), size=max_windows, replace=False, p=weights)
    return [index[int(i)] for i in chosen]


class PackedCrossSectionBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        arrays,
        index: list[tuple[int, int]],
        max_batch_windows: int,
        min_group_size: int,
        rng: np.random.Generator,
        drop_last: bool = False,
        batch_min_group_size: int | None = None,
    ) -> None:
        self.max_batch_windows = int(max_batch_windows)
        self.min_group_size = int(min_group_size)
        self.batch_min_group_size = int(batch_min_group_size if batch_min_group_size is not None else min_group_size)
        self.rng = rng
        self.drop_last = bool(drop_last)
        groups = _group_positions_by_datetime(arrays, index)
        self.groups = [pos for pos in groups.values() if len(pos) >= self.batch_min_group_size]
        self.groups.sort(key=lambda g: g[0])

    def __iter__(self):
        order = np.arange(len(self.groups))
        self.rng.shuffle(order)
        batch: list[int] = []
        for group_idx in order:
            group = list(self.groups[int(group_idx)])
            if batch and len(batch) + len(group) > self.max_batch_windows:
                yield batch
                batch = []
            if len(group) > self.max_batch_windows:
                for start in range(0, len(group), self.max_batch_windows):
                    chunk = group[start : start + self.max_batch_windows]
                    if len(chunk) == self.max_batch_windows or not self.drop_last:
                        yield chunk
            else:
                batch.extend(group)
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n_batches = 0
        cur = 0
        for group in self.groups:
            if len(group) > self.max_batch_windows:
                if cur:
                    n_batches += 1
                    cur = 0
                full, rem = divmod(len(group), self.max_batch_windows)
                n_batches += full
                if rem and not self.drop_last:
                    n_batches += 1
                continue
            if cur and cur + len(group) > self.max_batch_windows:
                n_batches += 1
                cur = 0
            cur += min(len(group), self.max_batch_windows)
        if cur and not self.drop_last:
            n_batches += 1
        return n_batches


def _cross_section_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    group_ids: torch.Tensor,
    loss_name: str,
    min_group_size: int,
) -> torch.Tensor:
    losses = []
    for gid in torch.unique(group_ids):
        mask = (group_ids == gid) & torch.isfinite(pred) & torch.isfinite(target)
        if int(mask.sum().item()) < min_group_size:
            continue
        p = pred[mask]
        y = target[mask]
        if loss_name == "ic":
            losses.append(ic_loss(p, y))
        elif loss_name == "centered_mse":
            p = (p - p.mean()) / (p.std(unbiased=False) + 1e-8)
            y = (y - y.mean()) / (y.std(unbiased=False) + 1e-8)
            losses.append(torch.nn.functional.mse_loss(p, y))
        else:
            raise ValueError(f"unknown cross-sectional loss: {loss_name}")
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def _group_mean_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    group_ids: torch.Tensor,
    min_group_size: int,
) -> torch.Tensor:
    losses = []
    for gid in torch.unique(group_ids):
        mask = (group_ids == gid) & torch.isfinite(pred) & torch.isfinite(target)
        if int(mask.sum().item()) < min_group_size:
            continue
        losses.append(torch.nn.functional.mse_loss(pred[mask].mean(), target[mask].mean()))
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def _group_centered_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    group_ids: torch.Tensor,
    min_group_size: int,
) -> torch.Tensor:
    losses = []
    for gid in torch.unique(group_ids):
        mask = (group_ids == gid) & torch.isfinite(pred) & torch.isfinite(target)
        if int(mask.sum().item()) < min_group_size:
            continue
        p = pred[mask]
        y = target[mask]
        losses.append(torch.nn.functional.mse_loss(p - p.mean(), y - y.mean()))
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset: WindowDataset,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    preds = []
    ys = []
    for xb, sym, time_ids, yb, _aux, _group_id in loader:
        xb = xb.to(device, non_blocking=True)
        sym = sym.to(device, non_blocking=True)
        time_ids = time_ids.to(device, non_blocking=True)
        out = _main_output(model(xb, sym, time_ids)).float().cpu().numpy()
        preds.append(out)
        ys.append(yb.numpy())
    pred = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    y = np.concatenate(ys) if ys else np.array([], dtype=np.float32)
    return {
        "loss_proxy_ic": compute_ic(pred, y),
        "loss_proxy_rank_ic": compute_rank_ic(pred, y),
    }


def train_split(
    config: dict[str, Any],
    arrays,
    feature_cols: list[str],
    split: RollingSplit,
    train_index: list[tuple[int, int]],
    val_index: list[tuple[int, int]],
    checkpoint_path: str | Path,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[EndToEndTransformerBaseline, float, float, dict[str, Any]]:
    train_cfg = config["train"]
    max_train = int(train_cfg.get("max_train_windows", 250_000))
    use_all_train = bool(train_cfg.get("use_all_train_windows", False))
    batch_mode = str(train_cfg.get("batch_mode", "window")).lower()
    cs_min_group_size = int(train_cfg.get("cs_min_group_size", 5))
    recency_halflife = float(train_cfg.get("sample_weight_halflife_days", 0.0) or 0.0)
    total_train_windows = len(train_index)
    total_val_windows = len(val_index)
    if not use_all_train and batch_mode == "cross_section" and len(train_index) > max_train:
        train_index = _limit_index_by_time_groups(arrays, train_index, max_train, cs_min_group_size, rng)
    elif not use_all_train and len(train_index) > max_train:
        if recency_halflife > 0:
            train_index = _sample_index_by_recency(arrays, train_index, max_train, recency_halflife, rng)
        else:
            chosen = rng.choice(len(train_index), size=max_train, replace=False)
            train_index = [train_index[int(i)] for i in chosen]
    max_val = int(train_cfg.get("max_val_windows", 50_000))
    use_all_val = bool(train_cfg.get("use_all_val_windows", False))
    if not use_all_val and len(val_index) > max_val:
        chosen = rng.choice(len(val_index), size=max_val, replace=False)
        val_index = [val_index[int(i)] for i in chosen]

    y = collect_targets(arrays, train_index)
    y = y[np.isfinite(y)]
    clip_tuple: tuple[float | None, float | None] = (None, None)
    winsor_q = float(train_cfg.get("label_winsor_q", 0.0) or 0.0)
    if winsor_q > 0:
        lo, hi = np.quantile(y, [winsor_q, 1.0 - winsor_q])
        y = np.clip(y, lo, hi)
        clip_tuple = (float(lo), float(hi))
    target_mean = float(np.nanmean(y))
    target_std = float(np.nanstd(y) + 1e-8)
    aux_cols = list(train_cfg.get("aux_targets") or [])
    aux_mean = np.zeros((0,), dtype=np.float32)
    aux_std = np.ones((0,), dtype=np.float32)
    aux_clip: tuple[np.ndarray | None, np.ndarray | None] = (None, None)
    if aux_cols:
        aux = collect_aux_targets(arrays, train_index)
        if aux.shape[1] != len(aux_cols):
            raise ValueError(f"aux target width {aux.shape[1]} does not match configured targets {len(aux_cols)}")
        aux_q = float(train_cfg.get("aux_winsor_q", 0.0) or 0.0)
        if aux_q > 0:
            lo = np.zeros(aux.shape[1], dtype=np.float32)
            hi = np.zeros(aux.shape[1], dtype=np.float32)
            aux_for_stats = aux.copy()
            for j in range(aux.shape[1]):
                vals = aux[:, j]
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    qlo, qhi = np.quantile(vals, [aux_q, 1.0 - aux_q])
                    lo[j], hi[j] = float(qlo), float(qhi)
                    aux_for_stats[:, j] = np.clip(aux_for_stats[:, j], lo[j], hi[j])
            aux_clip = (lo, hi)
            aux = aux_for_stats
        aux_mean_vals = []
        aux_std_vals = []
        for j in range(aux.shape[1]):
            vals = aux[:, j]
            vals = vals[np.isfinite(vals)]
            aux_mean_vals.append(float(np.mean(vals)) if len(vals) else 0.0)
            aux_std_vals.append(float(np.std(vals) + 1e-8) if len(vals) else 1.0)
        aux_mean = np.asarray(aux_mean_vals, dtype=np.float32)
        aux_std = np.asarray(aux_std_vals, dtype=np.float32)
    train_ds = WindowDataset(
        arrays,
        train_index,
        config["data"]["seq_len"],
        target_mean,
        target_std,
        target_clip=clip_tuple,
        aux_mean=aux_mean,
        aux_std=aux_std,
        aux_clip=aux_clip,
    )
    val_ds = (
        WindowDataset(
            arrays,
            val_index,
            config["data"]["seq_len"],
            target_mean,
            target_std,
            aux_mean=aux_mean,
            aux_std=aux_std,
            aux_clip=aux_clip,
        )
        if val_index
        else None
    )
    num_workers = int(train_cfg.get("num_workers", 2))
    loader_common = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0 and train_cfg.get("prefetch_factor") is not None:
        loader_common["prefetch_factor"] = int(train_cfg.get("prefetch_factor"))

    if batch_mode == "cross_section":
        batch_sampler = PackedCrossSectionBatchSampler(
            arrays,
            train_index,
            max_batch_windows=int(train_cfg.get("batch_size", 256)),
            min_group_size=cs_min_group_size,
            rng=rng,
            drop_last=bool(train_cfg.get("drop_last", False)),
            batch_min_group_size=int(train_cfg.get("batch_min_group_size", cs_min_group_size)),
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            **loader_common,
        )
    else:
        loader = DataLoader(
            train_ds,
            batch_size=int(train_cfg.get("batch_size", 256)),
            shuffle=True,
            drop_last=bool(train_cfg.get("drop_last", True)),
            **loader_common,
        )
    first_epoch_iter = None
    if (
        num_workers > 0
        and device.type == "cuda"
        and bool(train_cfg.get("prestart_train_workers", True))
        and loader_common.get("persistent_workers", False)
    ):
        # Start worker processes before the model creates a CUDA context.
        # Forking after CUDA initialization can make workers inherit GPU memory.
        first_epoch_iter = iter(loader)
    model = make_model(config, len(feature_cols), len(arrays), feature_cols=feature_cols).to(device)
    _load_initial_state(model, train_cfg, split)
    _apply_trainable_patterns(model, train_cfg)
    anchor_state = _make_anchor_state(model, train_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[baseline][{split.name}] model_params={n_params:,} trainable_params={n_trainable:,}",
        flush=True,
    )
    opt = _make_optimizer(model, train_cfg)
    epochs = int(train_cfg.get("epochs", 3))
    total_steps = max(1, epochs * len(loader) // max(1, int(train_cfg.get("grad_accum_steps", 1))))
    warmup = max(1, int(total_steps * float(train_cfg.get("warmup_frac", 0.08))))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and train_cfg.get("amp", True) else None
    grad_accum = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    best_val = -np.inf
    best_state = None
    history = []
    step = 0
    print(
        f"[baseline][{split.name}] train_windows={len(train_index)}/{total_train_windows} "
        f"val_windows={len(val_index)}/{total_val_windows} "
        f"target_std={target_std:.3e} aux_targets={len(aux_cols)} batch_mode={batch_mode} device={device}",
        flush=True,
    )
    aux_weight = float(train_cfg.get("aux_loss_weight", 0.0) or 0.0)
    feature_gate_l1 = float(train_cfg.get("feature_gate_l1", 0.0) or 0.0)
    operator_gate_l1 = float(train_cfg.get("operator_gate_l1", 0.0) or 0.0)
    operator_gate_smooth_l1 = float(train_cfg.get("operator_gate_smooth_l1", 0.0) or 0.0)
    operator_gate_binary_l1 = float(train_cfg.get("operator_gate_binary_l1", 0.0) or 0.0)
    cs_loss_weight = float(train_cfg.get("cs_loss_weight", 0.0) or 0.0)
    cs_loss_name = str(train_cfg.get("cs_loss", "ic")).lower()
    market_loss_weight = float(train_cfg.get("market_loss_weight", 0.0) or 0.0)
    decomp_market_weight = float(train_cfg.get("decomp_market_loss_weight", 0.0) or 0.0)
    decomp_residual_weight = float(train_cfg.get("decomp_residual_loss_weight", 0.0) or 0.0)
    moe_balance_weight = float(train_cfg.get("moe_balance_weight", 0.0) or 0.0)
    cvar_loss_weight = float(train_cfg.get("cvar_loss_weight", 0.0) or 0.0)
    cvar_tail_frac = float(train_cfg.get("cvar_tail_frac", 0.2) or 0.2)
    loss_recency_halflife = float(train_cfg.get("loss_recency_halflife_days", 0.0) or 0.0)
    loss_recency_min_weight = float(train_cfg.get("loss_recency_min_weight", 0.0) or 0.0)
    loss_recency_max_weight = float(train_cfg.get("loss_recency_max_weight", 0.0) or 0.0)
    cosine_ic_weight = float(train_cfg.get("cosine_ic_loss_weight", 0.0) or 0.0)
    pred_mean_penalty = float(train_cfg.get("pred_mean_penalty", 0.0) or 0.0)
    init_anchor_weight = float(train_cfg.get("init_anchor_weight", 0.0) or 0.0)
    moe_residual_output_l2 = float(train_cfg.get("moe_residual_output_l2", 0.0) or 0.0)
    eval_num_workers = int(train_cfg.get("eval_num_workers", train_cfg.get("num_workers", 2)))
    if bool(train_cfg.get("eval_initial_state", False)) and val_ds is not None and len(val_ds) > 0:
        val_stats = evaluate_model(
            model,
            val_ds,
            int(train_cfg.get("pred_batch_size", train_cfg.get("batch_size", 256))),
            device,
            num_workers=eval_num_workers,
        )
        if np.isfinite(val_stats["loss_proxy_ic"]) and val_stats["loss_proxy_ic"] > best_val:
            best_val = val_stats["loss_proxy_ic"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": 0, "loss": float("nan"), **val_stats})
        print(
            f"[baseline][{split.name}] epoch=0/initial val_ic={val_stats['loss_proxy_ic']:.4f}",
            flush=True,
        )
    for epoch in range(epochs):
        model.train()
        if bool(train_cfg.get("frozen_modules_eval", False)):
            _set_frozen_modules_eval(model)
        opt.zero_grad(set_to_none=True)
        losses = []
        accum = 0
        epoch_start = time.time()
        log_every = int(train_cfg.get("log_every_steps", 0) or 0)
        n_batches = len(loader)
        epoch_iter = first_epoch_iter if first_epoch_iter is not None else iter(loader)
        first_epoch_iter = None
        for xb, sym, time_ids, yb, auxb, group_ids in epoch_iter:
            xb = xb.to(device, non_blocking=True)
            sym = sym.to(device, non_blocking=True)
            time_ids = time_ids.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            auxb = auxb.to(device, non_blocking=True)
            group_ids = group_ids.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                out = model(xb, sym, time_ids)
                pred = _main_output(out)
                loss_name = str(train_cfg.get("loss", "combo")).lower()
                pointwise = _pointwise_loss(loss_name, pred, yb)
                weights = _recency_loss_weights(
                    group_ids,
                    split.train_end,
                    loss_recency_halflife,
                    loss_recency_min_weight,
                    loss_recency_max_weight,
                )
                if pointwise is not None and weights is not None:
                    main_loss = (pointwise * weights).sum() / weights.sum().clamp_min(1e-6)
                elif pointwise is not None:
                    main_loss = pointwise.mean()
                else:
                    main_loss = _loss(loss_name, pred, yb)
                aux_pred = _aux_output(out)
                if aux_weight > 0 and aux_pred is not None and auxb.numel() > 0:
                    if aux_pred.shape[-1] != auxb.shape[-1]:
                        raise ValueError(f"aux output width {aux_pred.shape[-1]} != aux target width {auxb.shape[-1]}")
                    mask = torch.isfinite(aux_pred) & torch.isfinite(auxb)
                    aux_loss = (
                        torch.nn.functional.mse_loss(aux_pred[mask], auxb[mask])
                        if mask.any()
                        else main_loss.new_tensor(0.0)
                    )
                    loss = main_loss + aux_weight * aux_loss
                else:
                    loss = main_loss
                if cvar_loss_weight > 0:
                    loss = loss + cvar_loss_weight * _cvar_mse_loss(pred, yb, cvar_tail_frac)
                if cosine_ic_weight > 0:
                    loss = loss + cosine_ic_weight * _cosine_ic_loss(pred, yb)
                if pred_mean_penalty > 0:
                    loss = loss + pred_mean_penalty * pred.mean().pow(2)
                if moe_residual_output_l2 > 0 and hasattr(model, "moe_residual_output_penalty"):
                    loss = loss + moe_residual_output_l2 * model.moe_residual_output_penalty()
                if cs_loss_weight > 0:
                    loss = loss + cs_loss_weight * _cross_section_loss(
                        pred,
                        yb,
                        group_ids,
                        cs_loss_name,
                        cs_min_group_size,
                    )
                if market_loss_weight > 0:
                    loss = loss + market_loss_weight * _group_mean_loss(
                        pred,
                        yb,
                        group_ids,
                        cs_min_group_size,
                    )
                aux_pred = _aux_output(out)
                if decomp_market_weight > 0 and aux_pred is not None and aux_pred.shape[-1] >= 1:
                    loss = loss + decomp_market_weight * _group_mean_loss(
                        aux_pred[:, 0],
                        yb,
                        group_ids,
                        cs_min_group_size,
                    )
                if decomp_residual_weight > 0 and aux_pred is not None and aux_pred.shape[-1] >= 2:
                    loss = loss + decomp_residual_weight * _group_centered_mse_loss(
                        aux_pred[:, 1],
                        yb,
                        group_ids,
                        cs_min_group_size,
                    )
                if feature_gate_l1 > 0 and hasattr(model, "regularization_loss"):
                    loss = loss + feature_gate_l1 * model.regularization_loss()
                if operator_gate_l1 > 0 and hasattr(model, "operator_regularization_loss"):
                    loss = loss + operator_gate_l1 * model.operator_regularization_loss()
                if operator_gate_smooth_l1 > 0 and hasattr(model, "operator_smoothness_loss"):
                    loss = loss + operator_gate_smooth_l1 * model.operator_smoothness_loss()
                if operator_gate_binary_l1 > 0 and hasattr(model, "operator_binary_loss"):
                    loss = loss + operator_gate_binary_l1 * model.operator_binary_loss()
                if moe_balance_weight > 0 and hasattr(model, "moe_regularization_loss"):
                    loss = loss + moe_balance_weight * model.moe_regularization_loss()
                if init_anchor_weight > 0 and anchor_state:
                    loss = loss + init_anchor_weight * _anchor_l2_loss(model, anchor_state)
                loss = loss / grad_accum
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                accum = 0
                continue
            losses.append(float(loss.detach().cpu()) * grad_accum)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum += 1
            if accum < grad_accum:
                continue
            if scaler is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= scale_before
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
                opt.step()
                optimizer_stepped = True
            opt.zero_grad(set_to_none=True)
            if optimizer_stepped:
                sched.step()
                step += 1
            accum = 0
            if optimizer_stepped and log_every > 0 and step > 0 and step % log_every == 0:
                elapsed = max(time.time() - epoch_start, 1e-6)
                seen_batches = min(n_batches, max(1, len(losses)))
                samples_seen = seen_batches * int(train_cfg.get("batch_size", 256))
                print(
                    f"[baseline][{split.name}] epoch={epoch + 1}/{epochs} "
                    f"step={seen_batches}/{n_batches} "
                    f"samples~={samples_seen} loss={float(np.mean(losses[-min(len(losses), 100):])):.5f} "
                    f"sec={elapsed:.1f}",
                    flush=True,
                )
        val_stats = {"loss_proxy_ic": float("nan"), "loss_proxy_rank_ic": float("nan")}
        if val_ds is not None and len(val_ds) > 0:
            val_stats = evaluate_model(
                model,
                val_ds,
                int(train_cfg.get("pred_batch_size", train_cfg.get("batch_size", 256))),
                device,
                num_workers=eval_num_workers,
            )
            if np.isfinite(val_stats["loss_proxy_ic"]) and val_stats["loss_proxy_ic"] > best_val:
                best_val = val_stats["loss_proxy_ic"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch + 1, "loss": mean_loss, **val_stats})
        print(
            f"[baseline][{split.name}] epoch={epoch + 1}/{epochs} loss={mean_loss:.5f} "
            f"val_ic={val_stats['loss_proxy_ic']:.4f}",
            flush=True,
        )
    if bool(train_cfg.get("restore_best_model", True)) and best_state is not None:
        model.load_state_dict(best_state, strict=True)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "feature_cols": feature_cols,
            "split": asdict(split),
            "config": config,
            "history": history,
            "label_clip": clip_tuple,
            "aux_cols": aux_cols,
            "aux_mean": aux_mean,
            "aux_std": aux_std,
            "aux_clip": aux_clip,
            "aux_loss_weight": aux_weight,
        },
        checkpoint_path,
    )
    return model, target_mean, target_std, {
        "history": history,
        "train_windows": len(train_index),
        "val_windows": len(val_index),
        "aux_cols": aux_cols,
        "aux_mean": aux_mean.tolist(),
        "aux_std": aux_std.tolist(),
    }


@torch.no_grad()
def predict_index(
    model: torch.nn.Module,
    arrays,
    pred_index: list[tuple[int, int]],
    seq_len: int,
    target_mean: float,
    target_std: float,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    destandardize: bool = True,
    output_mode: str = "main",
    aux_mean: np.ndarray | None = None,
    aux_std: np.ndarray | None = None,
    scale_aux_indices: list[int] | None = None,
    scale_power: float = 1.0,
    scale_min: float = 0.25,
    scale_max: float = 4.0,
) -> np.ndarray:
    ds = WindowDataset(arrays, pred_index, seq_len, target_mean, target_std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    model.eval()
    preds = []
    for xb, sym, time_ids, _yb, _aux, _group_id in loader:
        xb = xb.to(device, non_blocking=True)
        sym = sym.to(device, non_blocking=True)
        time_ids = time_ids.to(device, non_blocking=True)
        out = _prediction_output_from_model(
            model(xb, sym, time_ids),
            target_mean,
            target_std,
            destandardize=destandardize,
            output_mode=output_mode,
            aux_mean=aux_mean,
            aux_std=aux_std,
            scale_aux_indices=scale_aux_indices,
            scale_power=scale_power,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        preds.append(out.float().cpu().numpy().astype(np.float32))
    return np.concatenate(preds) if preds else np.array([], dtype=np.float32)
