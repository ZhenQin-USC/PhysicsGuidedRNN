# from socketserver import ForkingTCPServer
# from tkinter import font
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


def plot2(true, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    ax = np.array(axes)
    for i, ax1 in enumerate(ax.flat):
        # ax1 = axesf[i]
        if i < true.shape[1]:
            # ax1.set_ylim([0,1.05])
            ax1.plot(true[:, i], color='black')
        else:
            ax1.axis('off')


def plot2a(true, nrows, ncols, ylim=[0, 1.05]):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    ax = np.array(axes)
    for i, ax1 in enumerate(ax.flat):
        # ax1 = axesf[i]
        if i < true.shape[1]:
            ax1.set_ylim(ylim)
            ax1.plot(true[:, i], color='black')
        else:
            ax1.axis('off')


def plotpredictionlong2(true, predict0, frequency, trainwindow, nrows, ncols, ylim=[0, 1]):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    # plt.ylim(0.8,1)
    ax = np.array(axes)
    for i, ax1 in enumerate(ax.flat):
        if i < predict0.shape[-1]:
            # ax1.plot(true[:,i],color='black', linestyle = '-',marker = None,markersize = 5,label='OBS '+title[i])
            ax1.scatter(np.arange(0, true.shape[0], frequency), true[:, i], color='black', marker='o', s=1,
                        label='True')
            ax1.scatter(np.arange(0, predict0.shape[1] * frequency, frequency),
                        predict0[0, :, i], color='green', marker='o', s=1, label='Prediction')
            ax1.axvline(x=trainwindow, color='grey', linestyle='--')
            # ax1.set_title(title[i])
            # ax1.set_title('RMSE = ' + str(err))
            ax1.legend(loc=0, fontsize=8)
            ax1.set_ylim(ylim)
        else:
            ax1.axis('off')


def my_qqplot(ytrue, ypred, nstep=4, decimals=-1, step=None, Normalize=None):
    from matplotlib.ticker import StrMethodFormatter
    my_ceil = lambda x, decimals: np.ceil(x.max() * (10 ** decimals)) / (10 ** decimals)
    my_floor = lambda x, decimals: np.floor(x.min() * (10 ** decimals)) / (10 ** decimals)
    if np.ndim(ytrue) != 3:
        nlines = 1
        if np.ndim(ytrue) == 2:
            ytrue = np.expand_dims(ytrue, axis=2)
            ypred = np.expand_dims(ypred, axis=2)
        elif np.ndim(ytrue) == 1:
            ytrue = ytrue[:, None, None]  # np.expand_dims(ytrue, axis=(1,2))
            ypred = ypred[:, None, None]  # np.expand_dims(ypred,axis=(1,2))
    else:
        nlines = ytrue.shape[2]

    r2s = r2_score(np.reshape(ytrue, (-1, 1)), np.reshape(ypred, (-1, 1)))
    y = np.concatenate((ytrue, ypred), axis=1)
    delta = 10 ** (-decimals)
    vmax = my_ceil(y, decimals)
    vmin = my_floor(y, decimals)
    if step is None:
        step = my_ceil(np.abs(vmax - vmin) / nstep, decimals)
    vmax = vmin + nstep * step
    npoints = np.reshape(ytrue[:, :, 0], (-1, 1)).shape[0]
    dotsize = 1 * ((600 / npoints) ** 2)
    plt.figure(figsize=(3.9, 3.5), dpi=500)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
              '#7f7f7f', '#bcbd22', '#17becf', '#00FFFF', '#13EAC9', '#800000', '#FFA500']
    for i in range(nlines):
        sc = plt.scatter(np.reshape(ytrue[:, :, i], (-1, 1)),
                         np.reshape(ypred[:, :, i], (-1, 1)),
                         s=dotsize,
                         marker='.',
                         facecolor=colors[i])
    plt.xlim([vmin - delta, vmax + delta / 10])
    plt.ylim([vmin - delta, vmax + delta / 10])
    if Normalize:
        Ticks = np.linspace(0, 1, np.int32(nstep) + 1)
        Ticks = ["%.2f" % elem for elem in Ticks.tolist()]
        plt.yticks(np.arange(vmin, vmax + delta, step=step), Ticks)
        plt.xticks(np.arange(vmin, vmax + delta, step=step), Ticks)
    else:
        plt.yticks(np.arange(vmin, vmax + delta, step=step))
        plt.xticks(np.arange(vmin, vmax + delta, step=step))

    plt.title('$R^2$ Score: {0:.4f}'.format(r2s))
    plt.tight_layout()


def heatmap(data, xTickLabels, yTickLabels, xLabel, yLabel, fs=14, ax=None, splitLW=1,
            cbar_kw={}, cbarlabel="", showText=False,
            threshold=None, textcolors=("white", "black"), valfmt="{x:.1f}", **textkw):
    if not ax:
        ax = plt.gca()
    im = ax.imshow(data) #
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(xTickLabels)))
    ax.set_yticks(np.arange(len(yTickLabels)))
    # ... and label them with the respective list entries
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.set_xticklabels(xTickLabels, fontsize=fs, rotation=45)
    ax.set_yticklabels(yTickLabels, fontsize=fs)#
    ax.grid(which="minor", color="w", linestyle='-', linewidth=splitLW)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar.ax.tick_params(labelsize=fs)
    ax.set_xlabel(xLabel, fontsize=fs)
    ax.set_ylabel(yLabel, fontsize=fs)
    # Loop over data dimensions and create text annotations.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    if showText:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

