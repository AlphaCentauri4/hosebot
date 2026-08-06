#run after the analyze_opencvtriangle.py has run, as it requires the _synchronized.csv to work

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd
import matplotlib
import cmcrameri.cm as cmc
import matplotlib.colors as colors
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# Project layout, resolved relative to this Python file.
SCRIPT_DIR = Path(__file__).resolve().parent
#FILES_DIR = SCRIPT_DIR / "files"
VIDEOS_DIR = SCRIPT_DIR / "videos"
FIGURES_DIR = SCRIPT_DIR / "figures"


def format_axes(axes):
    for ax in axes:
        ax.xaxis.set_tick_params(labelsize=ticks_size)
        ax.yaxis.set_tick_params(labelsize=ticks_size)
        ax.grid(False)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
ticks_size = 14

experiment = "20260803_180357_10_20_7.5_glue"
dfname = experiment+"_synchronized.csv"
df = pd.read_csv("figures/"+dfname)
print(df.columns)
df = df[df.controller_time>170]
normalizedflow = (df.flow1-df.flow2)/(df.flow1+df.flow2)

#Detect maximum value index and use it for detecting time
max_idx = np.where(normalizedflow==np.max(normalizedflow))[0][0]
#df = df[df.controller_time<df.controller_time.iloc[max_idx]]


normalizedflow = (df.flow1-df.flow2)/(df.flow1+df.flow2)
def sigmoid(x, y_low, y_high, x0, k):
    z = np.clip(-k * (x - x0), -700, 700)
    return y_low + (y_high - y_low) / (1.0 + np.exp(z))


# Convert to aligned NumPy arrays
x = normalizedflow.to_numpy(dtype=float)
y = df["tip_transverse_mm"].to_numpy(dtype=float)

# Remove invalid observations
valid = np.isfinite(x) & np.isfinite(y)
x = x[valid]
y = y[valid]

# Initial estimates
p0 = [
    np.percentile(y, 1),   # lower asymptote
    np.percentile(y, 99),  # upper asymptote
    0.16,                  # transition location
    -30.0,                 # decreasing sigmoid
]

# Constrain the parameters to physically reasonable values
bounds = (
    [-5.0, -1.0, x.min(), -500.0],
    [-2.0,  1.0, x.max(),   -0.01],
)

params, covariance = curve_fit(
    sigmoid,
    x,
    y,
    p0=p0,
    bounds=bounds,
    maxfev=100_000,
)
print("y_low, y_high, x0, k", params)
# Evaluate the fitted model on an ordered grid
x_fit = np.linspace(x.min(), x.max(), 1000)
y_fit = sigmoid(x_fit, *params)



fig = plt.figure(figsize=(8, 7))
ax0 = fig.add_subplot(111)

ax0.set_title(
    '20260803_180357_10_20_7.5_glue',
    fontsize=ticks_size,
)

axes = [ax0]
format_axes(axes)

ax0.scatter(x, y, s=8, alpha=0.7, color='k')
ax0.plot(x_fit, y_fit, linewidth=1, label="Sigmoid fit",color='r')
ax0.legend(fontsize=ticks_size)

ax0.set_xlabel(
    "$\\frac{Q_{\\mathrm{left}}-Q_{\\mathrm{right}}}{Q_{\\mathrm{left}}+Q_{\\mathrm{right}}}$",
    fontsize=ticks_size,
)
ax0.set_ylabel(
    "Tip transverse displacement [mm]",
    fontsize=ticks_size,
)


plotname = f"figures/tip_normflow_calibration_{experiment}.png"

#Save fitting params
tipnormflow_calibrationvalues = (
    f"figures/tip_normflow_calibrationvalues_{experiment}.csv"
)
y_low, y_high, x0, k = params

pd.DataFrame({
    "y_low": [y_low],
    "y_high": [y_high],
    "x0": [x0],
    "k": [k],
}).to_csv(tipnormflow_calibrationvalues, index=False)

#and to call them back later
#cal = pd.read_csv(tipnormflow_calibrationvalues)

#y_low = cal.loc[0, "y_low"]
#y_high = cal.loc[0, "y_high"]
#x0 = cal.loc[0, "x0"]
#k = cal.loc[0, "k"]

#x_fit = np.linspace(x.min(), x.max(), 1000)
#y_fit = sigmoid(x_fit, *params)

y_low, y_high, x0, k = params

fit_text = (
    r"$y = y_{\mathrm{low}} + "
    r"\frac{y_{\mathrm{high}}-y_{\mathrm{low}}}"
    r"{1+\exp[-k(x-x_0)]}$"
    "\n"
    rf"$y_{{\mathrm{{low}}}} = {y_low:.4f}$"
    "\n"
    rf"$y_{{\mathrm{{high}}}} = {y_high:.4f}$"
    "\n"
    rf"$x_0 = {x0:.4f}$"
    "\n"
    rf"$k = {k:.4f}$"
)

ax0.text(
    0.98,
    0.98,
    fit_text,
    transform=ax0.transAxes,
    fontsize=11,
    va="top",
    ha="right",
    #bbox=dict(
    #    boxstyle="round,pad=0.5",
    #    facecolor="white",
    #    edgecolor="gray",
    #    alpha=0.9,
    #),
)

ax0.legend()
plt.tight_layout()


plt.savefig(
    plotname,
    dpi=200,
    bbox_inches="tight",
)

plt.show()
print(
    f"y_low={params[0]:.4f}, "
    f"y_high={params[1]:.4f}, "
    f"x0={params[2]:.4f}, "
    f"k={params[3]:.4f}"
)