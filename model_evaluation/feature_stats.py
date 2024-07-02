from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
from utilities import parse_col_name


def plot_distribution(ax, col, neg, pos, calc_log=False):
    ax.set_title(col, fontsize=15)
    ax.set_xlabel("values", fontsize=15)
    ylabel = 'log density' if calc_log else 'density'
    ax.set_ylabel(ylabel, fontsize=15)
    try:
        if len(neg) > 2:
            density = gaussian_kde(neg, bw_method=0.15)
            xs = np.linspace(min(neg), max(neg), 1000)
            d = np.log(density(xs)) if calc_log else density(xs)
            ax.plot(xs, d, color='#998ec3', lw=2.5)
        if len(pos) > 2:
            density = gaussian_kde(pos, bw_method=0.15)
            xs = np.linspace(min(pos), max(pos), 1000)
            d = np.log(density(xs)) if calc_log else density(xs)
            ax.plot(xs, d, color='#fdb863', lw=2.5)
        ax.legend(["Neg", "Pos"], loc="best", fontsize='large', prop={'size': 15})
    except np.linalg.LinAlgError:
        print(f"could not plot distribution for {col}")


def plot_attributes_distributions(X_train, y_train, attributes, n=3, save_figs=False):
    X_train = X_train.copy()
    for att in attributes[:n]:
        q = np.nanquantile(X_train[att], 0.95)
        X_train.loc[X_train[att] >= q, att] = np.nan

    positive = X_train[y_train != 0]
    negative = X_train[y_train == 0]
    fig = plt.figure(figsize=(25, 5*math.ceil(n/3)))
    for i in range(n):
        col = attributes[i]
        ax = fig.add_subplot(math.ceil(n/3), 3, i + 1)
        not_nan_pos = positive[pd.notnull(positive[col])][col].astype(float)
        not_nan_neg = negative[pd.notnull(negative[col])][col].astype(float)
        col = parse_col_name(col)
        plot_distribution(ax, col, not_nan_neg, not_nan_pos)

    fig.subplots_adjust(wspace=0.2, hspace=0.5)
    fig.savefig(f'top_{n}_features_dists.pdf') if save_figs else fig.show()