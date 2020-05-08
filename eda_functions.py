import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller


def dickey_fuller_df(df):
    """Takes in time series observations and conducts a Dickey-Fuller test for stationarity, returning the results in a DataFrame."""

    test = adfuller(df)
    dfoutput = pd.Series(test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in test[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


def rolling_ts_plot(df, halflife, figsize=(15, 10), style="darkgrid"):
    """Creates a single plot of time series observations combined with exponentially weighted rolling averages of the mean and standard
    deviations."""

    sns.set_style(style)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    mean = df.ewm(halflife=halflife).mean()
    std = df.ewm(halflife=halflife).std()
    df.plot(ax=ax)
    mean.plot(ax=ax)
    std.plot(ax=ax)
    ax.legend(["Original", "Exponentially Weighted Rolling Mean", "Exponentially Weighted Rolling Standard Deviation"]);
    plt.show()


def stationarity_check(df, halflife, figsize=(15, 10)):
    """A station that plots the rolling mean and standard deviation to the raw time series data and conducts a Dickey-Fuller
    test of the data returning the dataframe."""

    rolling_ts_plot(df, halflife, figsize=figsize)
    return dickey_fuller_df(df)


def plot_sarimax_one_step(model_results, observations, pred_date, date_trim=None, inv_func=None):
    """This is the code that plotted our graphs for our Sarimax findings. It is far from optimized and would need
    various edits for reuasability."""

    if not date_trim:
        date_trim = train.index[0]
    pred = model_results.get_prediction(start=pd.to_datetime(pred_date), dynamic=False)
    if inv_func:
        pred_ci = inv_func(pred.conf_int())
        pred_mean = inv_func(pred.predicted_mean)
        observations = inv_func(observations)
    else:
        pred_ci = pred.conf_int()
        pred_mean = pred.predicted_mean
    ax = observations[date_trim:].plot(label='observed')
    pred_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.legend()
    plt.show()

    actual = observations[pred_date:]
    mse = ((pred_mean - actual) ** 2).mean()
    print(f'Mean Squared Error - {mse}')
    print(f'Root Mean Squared Error - {math.sqrt(mse)}')

def dynamic_heatmap(df, columns, fontsize=20, annot=False, palette=None, figsize=(15, 10), squaresize=500):
    """Plots a heatmap that changes size values depending on correlation Adapted from:
    https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec"""

    plt.figure(figsize=figsize)
    corr = df[columns].corr()
    sns.set(style="dark")
    grid_bg_color = sns.axes_style()['axes.facecolor']

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    corr = pd.melt(corr.reset_index(),
                   id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']

    x = corr['x']
    y = corr['y']
    size = corr['value'].abs()

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right');

    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    size_scale = squaresize

    if palette:
        n_colors = len(palette)
    else:
        n_colors = 256  # Use 256 colors for the diverging color palette
        palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
    color_min, color_max = [-1,
                            1]  # Range of values that will be mapped to the palette, i.e. min and max possible correlation
    color = corr["value"]

    def value_to_color(val):
        val_position = float((val - color_min)) / (
                    color_max - color_min)  # position of value in the input range, relative to the length of the input range
        ind = int(val_position * (n_colors - 1))  # target index in the color palette
        return palette[ind]

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)  # Setup a 1x15 grid
    ax = plt.subplot(plot_grid[:, :-1])  # Use the leftmost 14 columns of the grid for the main plot

    ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=size * size_scale,  # Vector of square sizes, proportional to size parameter
        c=color.apply(value_to_color),  # Vector of square colors, mapped to color palette
        marker='s'  # Use square as scatterplot marker
    )

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    #     ax.set_fontsize(font_scale)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

    numbers = corr['value'].round(decimals=2)

    if annot:
        for i, txt in enumerate(numbers):
            annot_font_size = int(fontsize * size[i] * annot)
            ax.annotate(txt, (x.map(x_to_num)[i], y.map(x_to_num)[i]),
                        horizontalalignment="center", verticalalignment="center",
                        color=grid_bg_color, fontweight="black", fontsize=annot_font_size)

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    # Add color legend on the right side of the plot
    ax = plt.subplot(plot_grid[:, -1])  # Use the rightmost column of the plot

    col_x = [0] * len(palette)  # Fixed x coordinate for the bars
    bar_y = np.linspace(color_min, color_max, n_colors)  # y coordinates for each of the n_colors bars

    bar_height = bar_y[1] - bar_y[0]
    ax.barh(
        y=bar_y,
        width=[5] * len(palette),  # Make bars 5 units wide
        left=col_x,  # Make bars start at 0
        height=bar_height,
        color=palette,
        linewidth=0
    )
    ax.set_xlim(1, 2)  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    ax.grid(False)  # Hide grid
    ax.set_xticks([])  # Remove horizontal ticks
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vertical ticks for min, middle and max
    ax.yaxis.tick_right()  # Show vertical ticks on the right
    plt.show()