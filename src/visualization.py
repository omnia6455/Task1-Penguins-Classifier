import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

_COLORS = {1: '#1f77b4', -1: '#ff7f0e'}

def plot_decision_boundary(X_train, y_train, X_test, y_test, model, feature1_name, feature2_name, class1_name='Class 1 (+1)', class2_name='Class 2 (-1)', title=None, ax=None):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    all_x = np.concatenate([X_train[:, 0], X_test[:, 0]])
    all_y = np.concatenate([X_train[:, 1], X_test[:, 1]])
    x_min, x_max = all_x.min() - 1.0, all_x.max() + 1.0
    y_min, y_max = all_y.min() - 1.0, all_y.max() + 1.0

    W1, W2, b = model.boundary_coeffs()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = W1 * xx + W2 * yy + b
    ax.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], colors=['#ffe6cc', '#cce5ff'], alpha=0.6)
    for label, name in ((1, class1_name), (-1, class2_name)):
        tr = X_train[y_train == label]
        te = X_test[y_test == label]
        ax.scatter(tr[:, 0], tr[:, 1], c=_COLORS[label], marker='o', edgecolors='white', linewidths=1.0, s=75, label=f'Train: {name}', zorder=3)
        ax.scatter(te[:, 0], te[:, 1], c=_COLORS[label], marker='^', facecolors='none', edgecolors=_COLORS[label], linewidths=2.0, s=85, label=f'Test: {name}', zorder=3)

    W1, W2, b = model.boundary_coeffs()
    if abs(W2) > 1e-12:
        xs = np.linspace(x_min, x_max, 100)
        ys = -(W1 * xs + b) / W2
        ax.plot(xs, ys, color='#2c3e50', linewidth=2.5, linestyle='-', label='Decision boundary', zorder=2)
    else:
        x0 = -b / W1
        ax.axvline(x0, color='#2c3e50', linewidth=2.5, linestyle='-', label='Decision boundary', zorder=2)
    eq = f'{W1:.4f}*{feature1_name} + {W2:.4f}*{feature2_name} + {b:.4f} = 0'
    ax.set_title(title if title else 'Decision boundary', fontsize=13, fontweight='bold', pad=15)
    ax.text(0.02, 0.02, eq, transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc', boxstyle='round,pad=0.5'), zorder=4)

    ax.set_xlabel(feature1_name, fontsize=11)
    ax.set_ylabel(feature2_name, fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--', color='gray', zorder=1)

    return fig, ax

def plot_convergence(model, algorithm_name, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    if algorithm_name == 'Adaline':
        history = np.asarray(getattr(model, 'mse_history', []), dtype=float)
        ylabel = 'Mean squared error'
        color = '#2ca02c'
    else:
        history = np.asarray(getattr(model, 'errors', []), dtype=float)
        ylabel = 'Misclassified samples'
        color = '#d62728'

    epochs = range(1, len(history) + 1)
    ax.plot(epochs, history, '-o', markersize=4, color=color, linewidth=2, zorder=3)

    if len(history) > 0:
        ax.fill_between(epochs, history, alpha=0.1, color=color, zorder=2)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'{algorithm_name} convergence ({len(history)} epochs)', fontsize=13, fontweight='bold', pad=15)
    ax.grid(alpha=0.3, linestyle='--', color='gray', zorder=1)

    return fig, ax

def save_figure(fig, folder, filename, dpi=150):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    return path