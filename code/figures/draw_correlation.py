'''
This program is to generte the base figure of correlation analysis figure
Author: Yu Jiang
Last update: 2019-03-25

'''

import os
import sys
import numpy as np
import pandas as pd
import os.path as osp
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path

# set the program directory as working directory
program_dir = osp.dirname(osp.abspath(__file__))
os.chdir(program_dir)

# set default parameters for figure
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['font.weight'] = 'bold'

# set style for text and markers
outlier_prop = dict(markerfacecolor='r', marker='D', markersize=3)
median_val_prop = {'linewidth': 1, 'color': 'k'}
boxplot_labels = ['Control', 'Diuron']
boxplot_patch_colors = ['lightgreen', 'pink']
axis_font_prop = {'fontname': 'Arial', 'fontsize': 8, 'weight': 'bold'}
sig_font_prop = {'fontname': 'Arial',
                 'fontsize': 10, 'weight': 'bold', 'ha': 'center'}
marker_font_prop = {'fontname': 'Arial',
                    'fontsize': 12, 'weight': 'bold', 'ha': 'center'}


def main(base, model, group, th):
    main_folder = Path(os.getcwd()).parent / 'results' /'journal' / 'correlation_results' / base / model / group / f'th_{th}'
    value_filename = 'value.csv'
    pval_filename = 'pval.csv'
    corr_filename = 'corr.csv'

    value_df = pd.read_csv(main_folder / value_filename, sep=',')
    corr_df = pd.read_csv(main_folder / corr_filename, sep=',')
    corr_pval_df = pd.read_csv(main_folder / pval_filename, sep=',')

    g = sns.PairGrid(value_df, height=1, aspect=1)
    g = g.map_diag(plt.hist)
    g = g.map_offdiag(plt.scatter, s=3)

    xtick_list = list()
    ytick_list = list()

    # remove all labels and ticks
    for i, j in zip(*np.tril_indices_from(g.axes, 0)):
        # plt.setp(g.axes[i, j].get_xticklabels(), visible=False)
        #     g.axes[i, j].get_xaxis().get_label().set_visible(False)
        # plt.setp(g.axes[i, j].get_yticklabels(), visible=False)
        #     g.axes[i, j].get_yaxis().get_label().set_visible(False)
        plt.setp(g.axes[i, j].xaxis.get_majorticklabels(), rotation=90)
        cur_xticks = g.axes[i, j].get_xticks()
        cur_yticks = g.axes[i, j].get_yticks()
        xtick_list.append(cur_xticks)
        ytick_list.append(cur_yticks)

    # adjust the upper triangle figures
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        # g.axes[i, j].set_visible(False)
        g.axes[i, j].set_xlim(-1200, -1000, emit=False)
        g.axes[i, j].set_ylim(-1200, -1000, emit=False)
        if corr_pval_df.iloc[i, j] < 0.05:
            g.axes[i, j].text(-1100, -1100, '{0:.2f}'.format(
                corr_df.iloc[i, j]), color='r', **sig_font_prop)
        else:
            g.axes[i, j].text(-1100, -1100, '{0:.2f}'.format(
                corr_df.iloc[i, j]), color='k', **sig_font_prop)

    # add tick indicators for lower triangle figures
    tick_id = 0
    for i, j in zip(*np.tril_indices_from(g.axes, 0)):
        cur_xticks = g.axes[i, j].set_xticks(xtick_list[tick_id][1:-1])
        if i == 10 and j == 0:
            g.axes[i, j].set_ylim(-0.02, 0)
            g.axes[i, j].set_yticks([-0.015, -0.005])
        tick_id = tick_id + 1

    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98,
                        top=0.99, wspace=0.15, hspace=0.25)
    # g = g.add_legend()
    # plt.show()
    # output_figure_path = osp.join('Figure8_base.pdf')
    g.fig.tight_layout()
    plt.savefig(main_folder / 'correlation.png', format='png',
                dpi=300, facecolor='w', edgecolor='k')
    plt.close()

if __name__ == "__main__":
    # main()

    base = ['seg']
    model = ['DeepLab']
    group = ['severity_rate']

    for b in base:
        for g in group:
            for m in model:
                for th in np.arange(0.1, 1, 0.1):
                    main(b, m, g, round(th, 1))
                # main(b, m, g, 'otsu')