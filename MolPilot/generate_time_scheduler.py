import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm

from core.config.config import Config
from core.datasets import get_dataset
from core.datasets.pl_data import FOLLOW_BATCH
from core.models.sbdd4train import SBDD4Train, center_pos
import core.utils.transforms as trans


def build_dataloaders(cfg):
    if cfg.data.name == "pl_tr":
        dataset, subsets = get_dataset(config=cfg.data)
        train_set, test_set = subsets["train"], subsets["test"]
        cfg.dynamics.protein_atom_feature_dim = dataset.protein_atom_feature_dim
        cfg.dynamics.ligand_atom_feature_dim = dataset.ligand_atom_feature_dim
    else:
        protein_featurizer = trans.FeaturizeProteinAtom()
        ligand_featurizer = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
        transform = Compose(
            [
                protein_featurizer,
                ligand_featurizer,
                trans.FeaturizeLigandBond(),
            ]
        )
        cfg.dynamics.protein_atom_feature_dim = protein_featurizer.feature_dim
        cfg.dynamics.ligand_atom_feature_dim = ligand_featurizer.feature_dim
        cfg.dynamics.ligand_atom_type_dim = ligand_featurizer.type_feature_dim
        cfg.dynamics.ligand_atom_charge_dim = ligand_featurizer.charge_feature_dim
        cfg.dynamics.ligand_atom_aromatic_dim = ligand_featurizer.aromatic_feature_dim
        dataset, subsets = get_dataset(config=cfg.data, transform=transform)
        train_set, test_set = subsets["train"], subsets["test"]

    if "val" in subsets and len(subsets["val"]) > 0:
        val_set = subsets["val"]
    elif "valid" in subsets and len(subsets["valid"]) > 0:
        val_set = subsets["valid"]
    else:
        val_set = test_set

    collate_exclude_keys = ["ligand_nbh_list"]
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys,
    )
    return train_loader, val_loader


def _set_nested_attr(obj, key, value):
    if isinstance(value, dict):
        target = getattr(obj, key, None)
        if target is None:
            setattr(obj, key, value)
            return
        for sub_key, sub_value in value.items():
            _set_nested_attr(target, sub_key, sub_value)
    else:
        setattr(obj, key, value)


def load_checkpoint_metadata(ckpt_path, map_location="cpu"):
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    return checkpoint, state_dict, hyper_parameters


def apply_checkpoint_config_overrides(cfg, hyper_parameters, state_dict=None):
    if isinstance(hyper_parameters, dict):
        if "data" in hyper_parameters:
            data_cfg = hyper_parameters["data"]
            if (
                isinstance(data_cfg, dict)
                and "transform" in data_cfg
                and isinstance(data_cfg["transform"], dict)
                and "ligand_atom_mode" in data_cfg["transform"]
            ):
                cfg.data.transform.ligand_atom_mode = data_cfg["transform"]["ligand_atom_mode"]

        if "dynamics" in hyper_parameters and isinstance(hyper_parameters["dynamics"], dict):
            for key, value in hyper_parameters["dynamics"].items():
                _set_nested_attr(cfg.dynamics, key, value)

    # Fallback: infer critical dimensions from checkpoint weights when hyperparameters
    # are missing or incomplete.
    if state_dict is not None:
        v_head_bias = state_dict.get("dynamics.v_inference.2.bias", None)
        ligand_emb_weight = state_dict.get("dynamics.ligand_atom_emb.weight", None)
        if v_head_bias is not None:
            ckpt_feature_dim = int(v_head_bias.shape[0])
            cfg.dynamics.ligand_atom_feature_dim = ckpt_feature_dim
        else:
            ckpt_feature_dim = None

        if ligand_emb_weight is not None and ckpt_feature_dim is not None:
            ckpt_input_dim = int(ligand_emb_weight.shape[1])
            ckpt_time_emb_dim = ckpt_input_dim - ckpt_feature_dim
            if ckpt_time_emb_dim < 0:
                raise ValueError(
                    f"Invalid checkpoint dimensions: input_dim={ckpt_input_dim}, "
                    f"feature_dim={ckpt_feature_dim}"
                )
            cfg.dynamics.time_emb_dim = ckpt_time_emb_dim
            cfg.dynamics.adaptive_norm = ckpt_time_emb_dim == 0

        total_known = (
            int(getattr(cfg.dynamics, "ligand_atom_type_dim", 0))
            + int(getattr(cfg.dynamics, "ligand_atom_charge_dim", 0))
            + int(getattr(cfg.dynamics, "ligand_atom_aromatic_dim", 0))
        )
        if ckpt_feature_dim is not None and total_known != ckpt_feature_dim:
            type_dim = int(getattr(cfg.dynamics, "ligand_atom_type_dim", 0))
            charge_dim = int(getattr(cfg.dynamics, "ligand_atom_charge_dim", 0))
            aromatic_dim = int(getattr(cfg.dynamics, "ligand_atom_aromatic_dim", 0))

            if type_dim > 0 and charge_dim == 0 and aromatic_dim == 0 and ckpt_feature_dim == type_dim + 2:
                # Common MolPilot variant: `add_aromatic` atom types plus a separate
                # 2-class aromatic head.
                cfg.dynamics.ligand_atom_aromatic_dim = 2
            elif type_dim > 0 and charge_dim == 0 and aromatic_dim == 0 and ckpt_feature_dim == type_dim + 3:
                cfg.dynamics.ligand_atom_charge_dim = 3
            elif type_dim > 0 and charge_dim == 0 and aromatic_dim == 0 and ckpt_feature_dim == type_dim + 5:
                cfg.dynamics.ligand_atom_charge_dim = 3
                cfg.dynamics.ligand_atom_aromatic_dim = 2


def initialize_loss_grid(n_steps):
    loss_grid = {}
    for i in range(n_steps):
        loss_grid[i] = {}
        for j in range(n_steps):
            loss_grid[i][j] = {
                "pos": 0.0,
                "type": 0.0,
                "bond": 0.0,
                "pos_mse": 0.0,
                "type_ce": 0.0,
                "bond_ce": 0.0,
                "pos_cont": 0.0,
                "type_cont": 0.0,
                "bond_cont": 0.0,
                "loss": 0.0,
            }
    return loss_grid


def finalize_loss_grid(loss_grid, num_batches):
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    for i in loss_grid:
        for j in loss_grid[i]:
            for key in loss_grid[i][j]:
                loss_grid[i][j][key] /= num_batches
    return loss_grid


def build_loss_grid(model, loader, cfg, device, max_batches):
    pos_normalizer = torch.tensor(
        cfg.data.normalizer_dict.pos, dtype=torch.float32, device=device
    )
    n_steps = cfg.evaluation.sample_steps
    loss_grid = initialize_loss_grid(n_steps)
    total = n_steps * n_steps * (len(loader) if max_batches is None else min(len(loader), max_batches))
    progress = tqdm(total=total, desc="loss_grid")
    processed_batches = 0

    with torch.no_grad():
        model.eval()
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            processed_batches += 1
            batch = batch.to(device)
            protein_pos = getattr(batch, "protein_pos", None)
            protein_v = (
                batch.protein_atom_feature.float()
                if hasattr(batch, "protein_atom_feature")
                else None
            )
            batch_protein = getattr(batch, "protein_element_batch", None)
            ligand_pos = batch.ligand_pos / pos_normalizer
            ligand_v = batch.ligand_atom_feature_full
            batch_ligand = batch.ligand_element_batch

            if protein_pos is not None:
                protein_pos = protein_pos / pos_normalizer
                protein_pos, ligand_pos, _ = center_pos(
                    protein_pos,
                    ligand_pos,
                    batch_protein,
                    batch_ligand,
                    mode=cfg.dynamics.center_pos_mode,
                )
            else:
                ligand_pos = torch.zeros_like(ligand_pos)

            num_graphs = int(batch_ligand.max().item()) + 1
            for t_dis in range(n_steps):
                t_discrete = torch.full(
                    (num_graphs, 1),
                    (t_dis + 1) / n_steps,
                    dtype=ligand_pos.dtype,
                    device=device,
                )
                if not cfg.dynamics.use_discrete_t and not cfg.dynamics.destination_prediction:
                    t_discrete = torch.clamp(t_discrete, min=model.dynamics.t_min)

                for t_cont in range(n_steps):
                    t_pos = torch.full(
                        (num_graphs, 1),
                        (t_cont + 1) / n_steps,
                        dtype=ligand_pos.dtype,
                        device=device,
                    )
                    losses = model.dynamics.reconstruction_loss_one_step(
                        t_discrete,
                        protein_pos=protein_pos,
                        protein_v=protein_v,
                        batch_protein=batch_protein,
                        ligand_pos=ligand_pos,
                        ligand_v=ligand_v,
                        batch_ligand=batch_ligand,
                        ligand_bond_type=batch.ligand_fc_bond_type,
                        ligand_bond_index=batch.ligand_fc_bond_index,
                        batch_ligand_bond=batch.ligand_fc_bond_type_batch,
                        include_protein=model.include_protein,
                        t_pos=t_pos,
                        recon_loss=True,
                    )
                    pos_loss = losses["closs"]
                    type_loss = losses["dloss"]
                    bond_loss = losses["dloss_bond"]
                    total_loss = torch.mean(
                        pos_loss
                        + cfg.train.v_loss_weight * type_loss
                        + cfg.train.bond_loss_weight * bond_loss
                    )

                    cell = loss_grid[t_dis][t_cont]
                    cell["pos"] += float(pos_loss.mean())
                    cell["type"] += float(type_loss.mean())
                    cell["bond"] += float(bond_loss.mean())
                    cell["pos_mse"] += float(losses["closs_mse"].mean())
                    cell["type_ce"] += float(losses["dloss_ce"].mean())
                    cell["bond_ce"] += float(losses["dloss_bond_ce"].mean())
                    cell["pos_cont"] += float(losses["closs_cont"].mean())
                    cell["type_cont"] += float(losses["dloss_cont"].mean())
                    cell["bond_cont"] += float(losses["dloss_bond_cont"].mean())
                    cell["loss"] += float(total_loss.mean())
                    progress.update(1)

    progress.close()
    return finalize_loss_grid(loss_grid, processed_batches)


def build_cost_surface(loss_grid, loss_key="loss", clamp_max=20.0):
    n_steps = len(loss_grid)
    z = np.zeros((n_steps, n_steps), dtype=np.float32)
    for i in range(n_steps):
        for j in range(n_steps):
            z[i, j] = loss_grid[i][j][loss_key]
    z[z == 0] = clamp_max
    z = np.clip(z, 0, clamp_max)
    return z


def compute_dynamic_steps(z):
    grad_x, grad_y = np.gradient(z)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return 1.0 / (1.0 + grad_magnitude)


def advanced_flexible_dp_with_checks(z, budget, choose_closest=True):
    size = z.shape[0]
    dp = np.full((size, size, budget + 1), np.inf, dtype=np.float64)
    prev = np.full((size, size, budget + 1, 2), -1, dtype=np.int64)

    step_size = compute_dynamic_steps(z)
    step_size *= np.sqrt(2.0)
    step_size *= 2.0
    dp[0, 0, 0] = float(z[0, 0])

    for step in tqdm(range(budget), desc="dynamic_programming"):
        for x in range(size):
            for y in range(size):
                if not np.isfinite(dp[x, y, step]):
                    continue
                for kx in range(0, min(size - x, int(step_size[x, y]) + 2)):
                    for ky in range(0, min(size - y, int(step_size[x, y]) + 2)):
                        nx, ny = x + kx, y + ky
                        if nx >= size or ny >= size:
                            continue
                        if nx < x or ny < y:
                            continue
                        if kx + ky > step_size[x, y] + 1:
                            continue
                        if nx == 0 or ny == 0:
                            continue
                        if kx + ky == 0:
                            continue
                        cost = dp[x, y, step] + z[nx, ny]
                        if cost < dp[nx, ny, step + 1]:
                            dp[nx, ny, step + 1] = cost
                            prev[nx, ny, step + 1] = [x, y]

    final_step = int(np.argmin(dp[size - 1, size - 1]))
    if not np.isfinite(dp[size - 1, size - 1, final_step]):
        raise ValueError("No valid path found to the endpoint.")

    if choose_closest:
        closest_distance = abs(size - final_step)
        for step in range(budget, 0, -1):
            if np.isfinite(dp[size - 1, size - 1, step]):
                distance = abs(size - step)
                if distance < closest_distance:
                    final_step = step
                    closest_distance = distance

    path = []
    cx, cy, step = size - 1, size - 1, final_step
    while step >= 0:
        path.append((cx, cy))
        if cx == 0 and cy == 0:
            break
        cx, cy = prev[cx, cy, step]
        step -= 1

    if (cx, cy) != (0, 0):
        raise ValueError("Backtracking failed to reach the origin.")

    path.reverse()
    return np.asarray(path, dtype=np.int64), float(dp[size - 1, size - 1, final_step])


def load_model(cfg, state_dict, device):
    model = SBDD4Train(config=cfg)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="generate_scheduler")
    parser.add_argument("--revision", type=str, default="default")
    parser.add_argument("--mode", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--dp_budget", type=int, default=150)
    parser.add_argument("--max_batches", type=int, default=1)
    parser.add_argument("--loss_key", type=str, default="loss")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--version_tag", type=str, default="rect_closest")
    parser.add_argument("--ligand_atom_mode", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    return parser.parse_args()


def build_config(args):
    subs = {
        "exp_name": args.exp_name,
        "revision": args.revision,
        "debug": False,
        "no_wandb": True,
        "wandb_resume_id": None,
        "logging_level": "warning",
        "seed": args.seed,
        "test_only": True,
        "empty_folder": False,
        "ckpt_path": args.ckpt_path,
        "best_ckpt": "val_loss",
        "time_decoupled": False,
        "decouple_mode": "none",
        "skip_chem": False,
        "t_min": 0.0001,
        "sigma1_coord": 0.05,
        "beta1": 1.0,
        "beta1_bond": 1.0,
        "beta1_charge": 1.5,
        "beta1_aromatic": 3.0,
        "use_discrete_t": True,
        "discrete_steps": 1000,
        "destination_prediction": True,
        "sampling_strategy": "end_back_pmf",
        "time_emb_dim": 1,
        "time_emb_mode": "simple",
        "pos_init_mode": "zero",
        "bond_net_type": "lin",
        "pred_given_all": False,
        "pred_connectivity": False,
        "self_condition": False,
        "num_blocks": 1,
        "num_layers": 4,
        "hidden_dim": 128,
        "adaptive_norm": False,
        "ligand_atom_mode": args.ligand_atom_mode or "add_aromatic",
        "pos_normalizer": 2.0,
        "visual_chain": False,
        "batch_size": args.batch_size if args.batch_size is not None else 4,
        "pos_noise_std": 0.0,
        "random_rot": False,
        "epochs": 15,
        "resume": False,
        "v_loss_weight": 1.0,
        "bond_loss_weight": 10.0,
        "max_grad_norm": "Q",
        "lr": 5e-4,
        "weight_decay": 0.0,
        "scheduler": "plateau",
        "eval_batch_size": args.eval_batch_size if args.eval_batch_size is not None else 100,
        "sample_steps": args.sample_steps,
        "num_samples": 10,
        "sample_num_atoms": "ref",
        "ligand_path": None,
        "protein_path": None,
        "fix_bond": False,
        "mode": args.mode,
        "time_scheduler_path": None,
        "time_coef": 1.0,
    }
    return Config(args.config_file, **subs)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = build_config(args)
    cfg.test_only = True
    cfg.no_wandb = True
    cfg.evaluation.mode = args.mode
    cfg.evaluation.sample_steps = args.sample_steps
    cfg.accounting.test_outputs_dir = args.output_dir

    checkpoint, state_dict, hyper_parameters = load_checkpoint_metadata(args.ckpt_path)
    apply_checkpoint_config_overrides(cfg, hyper_parameters, state_dict=state_dict)

    if args.ligand_atom_mode is not None:
        cfg.data.transform.ligand_atom_mode = args.ligand_atom_mode
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.eval_batch_size is not None:
        cfg.evaluation.batch_size = args.eval_batch_size

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    train_loader, val_loader = build_dataloaders(cfg)
    apply_checkpoint_config_overrides(cfg, hyper_parameters, state_dict=state_dict)
    loader = train_loader if args.mode == "train" else val_loader

    print(
        f"Using device={device}, mode={args.mode}, sample_steps={args.sample_steps}, "
        f"max_batches={args.max_batches}"
    )
    print(
        "Feature dims:",
        f"protein={cfg.dynamics.protein_atom_feature_dim}",
        f"ligand_total={cfg.dynamics.ligand_atom_feature_dim}",
        f"ligand_type={getattr(cfg.dynamics, 'ligand_atom_type_dim', 'NA')}",
        f"ligand_charge={getattr(cfg.dynamics, 'ligand_atom_charge_dim', 'NA')}",
        f"ligand_aromatic={getattr(cfg.dynamics, 'ligand_atom_aromatic_dim', 'NA')}",
    )

    print(
        "Recovered checkpoint config:",
        f"ligand_mode={cfg.data.transform.ligand_atom_mode}",
        f"ligand_total={cfg.dynamics.ligand_atom_feature_dim}",
        f"ligand_type={getattr(cfg.dynamics, 'ligand_atom_type_dim', 'NA')}",
        f"ligand_charge={getattr(cfg.dynamics, 'ligand_atom_charge_dim', 'NA')}",
        f"ligand_aromatic={getattr(cfg.dynamics, 'ligand_atom_aromatic_dim', 'NA')}",
        f"time_emb_dim={getattr(cfg.dynamics, 'time_emb_dim', 'NA')}",
        f"adaptive_norm={getattr(cfg.dynamics, 'adaptive_norm', 'NA')}",
    )

    model = load_model(cfg, state_dict, device)
    loss_grid = build_loss_grid(
        model=model,
        loader=loader,
        cfg=cfg,
        device=device,
        max_batches=args.max_batches,
    )

    ckpt_stem = Path(args.ckpt_path).stem
    loss_grid_name = f"loss_grid_rect{args.sample_steps}_{args.mode}_{ckpt_stem}.pt"
    loss_grid_path = output_dir / loss_grid_name
    torch.save(loss_grid, loss_grid_path)
    print(f"Saved loss grid to {loss_grid_path}")

    z = build_cost_surface(loss_grid, loss_key=args.loss_key)
    optimized_path, path_cost = advanced_flexible_dp_with_checks(
        z=z,
        budget=args.dp_budget,
        choose_closest="closest" in args.version_tag,
    )

    scheduler_stem = f"optimized_path_{args.version_tag}_{args.mode}_{ckpt_stem}"
    scheduler_tensor = torch.from_numpy(optimized_path.astype(np.float32))
    scheduler_tensor_path = output_dir / f"{scheduler_stem}.pt"
    scheduler_numpy_path = output_dir / f"{scheduler_stem}_numpy_compat.pt"
    torch.save(scheduler_tensor, scheduler_tensor_path)
    torch.save(optimized_path, scheduler_numpy_path)
    print(f"Saved scheduler tensor to {scheduler_tensor_path}")
    print(f"Saved scheduler numpy-compatible file to {scheduler_numpy_path}")
    print(f"Path length={len(optimized_path)}, total_cost={path_cost:.6f}")


if __name__ == "__main__":
    main()
