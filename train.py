import os
import time
import argparse
import json
import random
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from trainer import *
from data import *
from plotter import Plotter

def parse_args():
    """
    Parse command-line arguments for training or inference.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including device, input files, hyperparameters, and output options.
    """
    parser = argparse.ArgumentParser(description="Transformer Autoencoder Training")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "auto"], default="auto",
                        help="Choose device: cpu, gpu, or auto (default: auto)")
    parser.add_argument("inputs", type=str, nargs="*", default=["hitsTracks.csv"],
                        help="One or more input CSV files")
    parser.add_argument("--max_epochs", type=int, default=120,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for DataLoader")
    parser.add_argument("--outdir", type=str, default="outputs/local",
                        help="Directory to save models and plots")
    parser.add_argument("--end_name", type=str, default="",
                        help="Optional suffix to append to output files")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="dimension of hidden layer")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads in the transformer")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of transformer encoder layers")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--no_train", action="store_true",
                        help="Skip training and only run inference using a saved model")
    parser.add_argument("--enable_progress_bar", action="store_true",
                        help="Enable progress bar during training (default: disabled)")
    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

# -----------------------------
def main():
    """
    Main script to train or test the Transformer Masked Autoencoder.

    Workflow:
        1. Parse arguments and create output directories.
        2. Load CSV data files and convert to FeatureDataset.
        3. Split dataset into training and validation sets.
        4. Initialize the TransformerAutoencoder model.
        5. Train the model if not skipped.
        6. Run inference on the validation set and plot results.
    """
    set_seed(42)

    args = parse_args()

    inputs = args.inputs if args.inputs else ["avgWires.csv"]
    outDir = args.outdir
    maxEpochs = args.max_epochs
    batchSize = args.batch_size
    end_name = args.end_name
    doTraining = not args.no_train
    os.makedirs(outDir, exist_ok=True)

    # -----------------------------
    print('\n\nLoading data...')
    startT_data = time.time()

    hits_all = []
    states_all = []
    for fname in inputs:
        print(f"Loading data from {fname} ...")
        hits_list, states = read_tracks_with_hits(fname)
        hits_all.extend(hits_list)  # list of [num_hits, 5]
        states_all.extend(states)  # list of [5]

    states_all = np.array(states_all, dtype=np.float32)

    # === hit & state stats path ===
    hit_stats_out_path = os.path.join(outDir, "hit_stats.json")
    state_stats_out_path = os.path.join(outDir, "state_stats.json")
    hit_stats_nets_path = os.path.join("nets", "hit_stats.json")
    state_stats_nets_path = os.path.join("nets", "state_stats.json")

    # === 加载或计算统计量 ===
    if args.no_train:
        # 推理模式：从 nets/ 加载
        print("\n=== 推理模式: 从 nets/ 读取统计量 ===")
        if not os.path.exists(hit_stats_nets_path) or not os.path.exists(state_stats_nets_path):
            raise FileNotFoundError("Normalization stats not found in nets/ directory.")
        with open(hit_stats_nets_path, "r") as f:
            hit_stats = json.load(f)
        with open(state_stats_nets_path, "r") as f:
            state_stats = json.load(f)
        print(f"Loaded normalization stats from nets/")
    else:
        # ---- 自动统计 hits 与 state 归一化参数 ----
        print("\n=== 自动计算归一化统计量 ===")
        all_hits = np.vstack(hits_all)
        doca_vals = all_hits[:, 0]
        xm_vals = all_hits[:, 1]
        xr_vals = all_hits[:, 2]
        yr_vals = all_hits[:, 3]
        z_vals = all_hits[:, 4]

        hit_stats = {
            "doca_mean": float(doca_vals.mean()),
            "doca_std": float(doca_vals.std()),
            "xm_mean": float(xm_vals.mean()),
            "xm_std": float(xm_vals.std()),
            "xr_mean": float(xr_vals.mean()),
            "xr_std": float(xr_vals.std()),
            "yr_mean": float(yr_vals.mean()),
            "yr_std": float(yr_vals.std()),
            "z_mean": float(z_vals.mean()),
            "z_std": float(z_vals.std()),
        }

    print(f"doca: mean={hit_stats['doca_mean']:.6g}, std={hit_stats['doca_std']:.6g}")
    print(f"xm: mean={hit_stats['xm_mean']:.6g}, std={hit_stats['xm_std']:.6g}")
    print(f"xr: mean={hit_stats['xr_mean']:.6g}, std={hit_stats['xr_std']:.6g}")
    print(f"yr: mean={hit_stats['yr_mean']:.6g}, std={hit_stats['yr_std']:.6g}")
    print(f"z   : mean={hit_stats['z_mean']:.6g}, std={hit_stats['z_std']:.6g}")

    state_names = ["vx", "vy", "tx", "ty", "Q"]
    state_stats = {}
    print("\n=== State 统计 ===")
    for i, name in enumerate(state_names):
        vals = states_all[:, i]
        mean, std = float(vals.mean()), float(vals.std())
        state_stats[name] = (mean, std)
        print(f"{name:>3s}: mean={mean:.6g}, std={std:.6g}")

    # 保存统计量到 outDir/
    os.makedirs(outDir, exist_ok=True)
    with open(hit_stats_out_path, "w") as f:
        json.dump(hit_stats, f, indent=2)
    with open(state_stats_out_path, "w") as f:
        json.dump(state_stats, f, indent=2)
    print(f"Saved normalization stats to {outDir}/")

    print("=========================================\n")

    # 初始化数据集（自动归一化）
    dataset = TrackDataset(
        hits_list=hits_all,
        states=states_all,
        normalize=True,
        hit_stats=hit_stats,
        state_stats=state_stats
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    print('\n\nTrain size:', train_size)
    print('Validation size:', val_size)

    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batchSize, shuffle=False, collate_fn=collate_fn)

    hits_sample, state_sample, mask_sample = next(iter(train_loader))
    print('hits_sample shape:', hits_sample.shape)  # e.g. [batch, num_hits, 5]
    print('state_sample shape:', state_sample.shape)  # e.g. [batch, 5]
    print('mask_sample shape:', mask_sample.shape)

    endT_data = time.time()
    print(f'Loading data took {endT_data - startT_data:.2f}s \n\n')


    # Define plotter
    plotter = Plotter(print_dir=outDir, end_name=end_name)

    # -----------------------------
    if args.hidden_dim % args.nhead != 0:
        raise ValueError(f"d_model ({args.hidden_dim}) must be divisible by nhead ({args.nhead})")

    model = TrackTransformer(
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        lr=args.lr
    )

    loss_tracker = LossTracker()

    # -----------------------------
    if doTraining:
        if args.device == "cpu":
            accelerator = "cpu"; devices = 1
        elif args.device == "gpu":
            if torch.cuda.is_available(): accelerator="gpu"; devices=1
            else: print("GPU not available. Falling back to CPU."); accelerator="cpu"; devices=1
        elif args.device == "auto":
            if torch.cuda.is_available(): accelerator="gpu"; devices="auto"
            else: accelerator="cpu"; devices=1
        else:
            raise ValueError(f"Unknown device option: {args.device}")

        print(f"Using accelerator={accelerator}, devices={devices}")

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy="auto",
            max_epochs=maxEpochs,
            enable_progress_bar=args.enable_progress_bar,
            log_every_n_steps=1000,
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            logger=False,
            callbacks=[loss_tracker]
        )

        print('\n\nTraining...')
        startT_train = time.time()
        trainer.fit(model, train_loader, val_loader)
        endT_train = time.time()
        print(f'Training took {(endT_train - startT_train)/60:.2f} minutes \n\n')

        plotter.plotTrainLoss(loss_tracker)

        # Save model
        # 先把模型切换到 CPU 并 eval 模式
        model.to("cpu")
        model.eval()

        # 包装模型，使 forward 自动生成全 False mask
        wrapper_model = TrackTransformerWrapper(model)
        wrapper_model.eval()

        # TorchScript 导出（script）
        torchscript_model = torch.jit.script(wrapper_model)

        # 保存
        torchscript_model.save(f"{outDir}/tae_{end_name}.pt")
    # -----------------------------
    # Load model for inference
    model_file = f"{outDir}/tae_{end_name}.pt" if doTraining else "nets/tae_default.pt"
    model = torch.jit.load(model_file)
    model.eval()

    val_loader2 = DataLoader(val_set, batch_size=1, shuffle=False)

    all_preds = []
    all_targets = []

    startT_test = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for hits_batch, state_batch in val_loader2:
            hits_batch = hits_batch.to(device)
            state_batch = state_batch.to(device)

            y_pred = model(hits_batch)
            y_true = state_batch

            all_preds.append(y_pred.cpu())
            all_targets.append(y_true.cpu())

    endT_test = time.time()
    print(f'Test with {len(val_loader2.dataset)} samples took {endT_test - startT_test:.2f}s \n\n')

    # 合并整个验证集
    all_preds = torch.cat(all_preds, dim=0).numpy()  # [N, 5]
    all_targets = torch.cat(all_targets, dim=0).numpy()  # [N, 5]

    # -----------------------------
    # ✅ 反归一化预测与真实值
    def denormalize_state(states, stats):
        """反归一化函数"""
        result = states.copy()
        for i, key in enumerate(["vx", "vy", "tx", "ty", "Q"]):
            mean, std = stats[key]
            result[:, i] = result[:, i] * std + mean
        return result

    all_preds_denorm = denormalize_state(all_preds, state_stats)
    all_targets_denorm = denormalize_state(all_targets, state_stats)

    # 使用 Plotter 绘图
    plotter.plot_diff(all_targets_denorm, all_preds_denorm)
    plotter.plot_pred_target(all_targets_denorm, all_preds_denorm)


if __name__ == "__main__":
    main()
