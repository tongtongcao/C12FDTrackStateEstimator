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
        Plot differences and histograms for track state predictions.

        Parameters
        ----------
        y_true : np.ndarray, shape [N,7]
            True track states: [q, px, py, pz, vx, vy, vz]
        y_pred : np.ndarray, shape [N,7]
            Predicted track states
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        names = ['q', 'px','py','pz','vx','vy','vz']

        # 1. plot differences for px,py,pz,vx,vy,vz
        fig, axes = plt.subplots(2, 3, figsize=(18,10))
        axes = axes.flatten()
        for i, name in enumerate(names[1:]):  # skip q
            diff = y_pred[:, i+1] - y_true[:, i+1]  # 注意：y_true[:,1:] 对应 px~vz
            axes[i].hist(diff, bins=50, color='royalblue', alpha=0.7)
            axes[i].set_title(f"{name} diff.")
            axes[i].set_xlabel("Difference")
            axes[i].set_ylabel("Counts")
        plt.tight_layout()
        plt.savefig(os.path.join(self.print_dir, f"track_diff_{self.end_name}.png"))
        plt.close()

        # 2. histogram for charge q
        q_true = y_true[:,0]
        q_pred = y_pred[:,0]
        plt.figure(figsize=(8,6))
        for q_val in [-1, 1]:
            mask = q_true == q_val
            plt.hist(q_pred[mask], bins=50, alpha=0.7, label=f"true q={q_val}")
        plt.xlabel("Predicted q")
        plt.ylabel("Counts")
        plt.title("Charge prediction histogram")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.print_dir, f"track_charge_{self.end_name}.png"))
        plt.close()

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
        components = ['px', 'py', 'pz', 'vx', 'vy', 'vz']
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