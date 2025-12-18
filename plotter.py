import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import norm

plt.rcParams.update({
    'font.size': 15,
    'legend.edgecolor': 'white',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size': 15,
    'xtick.minor.size': 10,
    'ytick.major.size': 15,
    'ytick.minor.size': 10,
    'xtick.major.width': 3,
    'xtick.minor.width': 3,
    'ytick.major.width': 3,
    'ytick.minor.width': 3,
    'axes.linewidth': 3,
    'figure.max_open_warning': 200,
    'lines.linewidth': 5
})

class Plotter:
    def __init__(self, print_dir='', end_name=''):
        """
        Parameters
        ----------
        print_dir : str
            Directory where plots will be saved.
        end_name : str
            Optional suffix appended to output file names.
        """
        self.print_dir = print_dir
        self.end_name = end_name

    def plotTrainLoss(self, tracker):
        """
        Plot training and validation loss curves.

        Parameters
        ----------
        tracker : object
            Object with attributes 'train_losses' and 'val_losses', containing per-epoch loss values.
        """
        train_losses = tracker.train_losses
        val_losses = tracker.val_losses

        plt.figure(figsize=(20, 20))
        plt.plot(train_losses, label='Train', color='royalblue')
        plt.plot(val_losses, label='Test', color='firebrick')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        outname = f"{self.print_dir}/loss_{self.end_name}.png"
        plt.savefig(outname)
        plt.close()

    def plot_diff(self, y_true, y_pred):
        """
        Plot differences (y_pred - y_true) with Gaussian fits and fixed fit/plot ranges.

        Parameters
        ----------
        y_true : np.ndarray, shape [N,7]
            True track states: [q, px, py, pz, vx, vy, vz]
        y_pred : np.ndarray, shape [N,7]
            Predicted track states
        show : bool, optional
            If True, display plots interactively
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        import pandas as pd

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have same shape"

        # === 预定义每个变量的拟合范围和显示范围 ===
        fit_ranges = {
            'px': (-0.05, 0.05),
            'py': (-0.05, 0.05),
            'pz': (-0.15, 0.15),
            'vx': (-1.0, 1.0),
            'vy': (-5.0, 5.0)
        }
        plot_ranges = {
            'px': (-0.2, 0.2),
            'py': (-0.2, 0.2),
            'pz': (-0.5, 0.5),
            'vx': (-5, 5),
            'vy': (-15, 15)
        }

        names = ['q', 'px', 'py', 'pz', 'vx', 'vy']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        results = []

        for i, name in enumerate(names[1:]):  # skip q
            diff = y_pred[:, i + 1] - y_true[:, i + 1]
            diff = diff[np.isfinite(diff)]

            # 取出固定范围
            fit_min, fit_max = fit_ranges[name]
            plot_min, plot_max = plot_ranges[name]

            # 拟合数据
            mask_fit = (diff >= fit_min) & (diff <= fit_max)
            diff_fit = diff[mask_fit]

            # 高斯拟合
            mu, sigma = norm.fit(diff_fit)
            results.append((name, mu, sigma))

            # 直方图
            n, bins, _ = axes[i].hist(
                diff, bins=60, range=(plot_min, plot_max),
                color='royalblue', alpha=0.7
            )

            # 拟合曲线
            x_fit = np.linspace(plot_min, plot_max, 400)
            y_fit = norm.pdf(x_fit, mu, sigma) * len(diff_fit) * (bins[1] - bins[0])
            axes[i].plot(x_fit, y_fit, 'r--', lw=3, label=f"μ={mu:+.3e}\nσ={sigma:.3e}")

            # ±1σ 阴影
            #axes[i].axvspan(mu - sigma, mu + sigma, color='gray', alpha=0.2, label='±1σ')

            axes[i].set_title(f"{name} diff")
            axes[i].set_xlabel("Difference")
            axes[i].set_ylabel("Counts")
            axes[i].legend(loc='upper left')
            #axes[i].grid(True, linestyle='--', alpha=0.4)
            axes[i].set_xlim(plot_min, plot_max)

        plt.tight_layout()
        outname = os.path.join(self.print_dir, f"track_diff_{self.end_name}.png")
        plt.savefig(outname, dpi=200)
        plt.close()

        # --- Charge 直方图 ---
        q_true = np.sign(y_true[:, 0]).astype(int)
        q_pred = y_pred[:, 0]
        plt.figure(figsize=(8, 6))
        for q_val in np.unique(q_true):
            mask = q_true == q_val
            plt.hist(q_pred[mask], bins=50, alpha=0.7, label=f"true q={q_val}")
        plt.xlabel("Predicted q")
        plt.ylabel("Counts")
        plt.title("Charge prediction histogram")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        outname = os.path.join(self.print_dir, f"track_charge_{self.end_name}.png")
        plt.savefig(outname, dpi=200)
        plt.close()

        # --- 保存拟合结果 ---
        df = pd.DataFrame(results, columns=['Variable', 'Mean(μ)', 'Sigma(σ)'])
        csv_path = os.path.join(self.print_dir, f"track_diff_fit_{self.end_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved fit results to {csv_path}")


    def plot_pred_target(self, targets, preds):
        """
        Plot predicted vs true values for px, py, pz, vx, vy, vz.

        Parameters
        ----------
        targets : np.ndarray, shape [N, 7]
            Ground truth track states (q, px, py, pz, vx, vy, vz)
        preds : np.ndarray, shape [N, 7]
            Model predictions
        """
        components = ['px', 'py', 'pz', 'vx', 'vy']
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, comp in enumerate(components):
            ax = axes[i]
            true_vals = targets[:, i + 1]  # +1 因为第0列是 charge q
            pred_vals = preds[:, i + 1]

            ax.scatter(true_vals, pred_vals, s=10, alpha=0.5, color='royalblue')
            ax.plot([true_vals.min(), true_vals.max()],
                    [true_vals.min(), true_vals.max()],
                    'r--', lw=2)
            ax.set_xlabel(f"True {comp}")
            ax.set_ylabel(f"Pred {comp}")
            ax.set_title(f"{comp}: Predicted vs True")
            ax.grid(True)

        plt.tight_layout()
        outname = f"{self.print_dir}/track_pred_vs_true_{self.end_name}.png"
        plt.savefig(outname)
        plt.close()