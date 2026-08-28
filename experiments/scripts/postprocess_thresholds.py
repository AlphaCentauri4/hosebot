from __future__ import annotations

from pathlib import Path
import argparse
import csv
import re
from typing import Optional

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
dpi = 300


filename = "transition_summary.csv"
df = pd.read_csv(filename)


idxcol_up_pavg, idxcol_up_pstd = 2, 4
idxcol_up_Qavg, idxcol_up_Qstd  = 3, 5

idxcol_down_pavg, idxcol_down_pstd = 6, 8
idxcol_down_Qavg, idxcol_down_Qstd  = 7, 9
print(len(df))

color_ed22, color_ed32 = '#483CB0', '#47B03C'
capsize = 0 
ticks_size = 14
fig = plt.figure(figsize=(12, 6))
ax0 = fig.add_subplot(132)
ax1 = fig.add_subplot(233)
ax2 = fig.add_subplot(236)
axes = [ax0,ax1,ax2]

for ax in axes:
    ax.xaxis.set_tick_params(labelsize=ticks_size)
    ax.yaxis.set_tick_params(labelsize=ticks_size)
    ax.grid(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
### ax0  
ax0.scatter(-10, -10, color='k', marker = '^', label = "Forward transition")
ax0.scatter(-10, -10, color='k', marker = 'v', label = "Backward transition")

ax0.scatter(-10, -10, color='white', edgecolor=color_ed22, marker = 'o', label = "ED22")
ax0.scatter(-10, -10, color='white',  edgecolor=color_ed32, marker = 'o', label = "ED32")

for idx in range(len(df)):#

	if df.iloc[idx][idxcol_up_pavg] != 0 :

		pupval_avg, pupval_std = df.iloc[idx][idxcol_up_pavg], df.iloc[idx][idxcol_up_pstd]
		pdownval_avg, pdownval_std = df.iloc[idx][idxcol_down_pavg], df.iloc[idx][idxcol_down_pstd]

		Qupval_avg, Qupval_std = df.iloc[idx][idxcol_up_Qavg], df.iloc[idx][idxcol_up_Qstd]
		Qdownval_avg, Qdownval_std = df.iloc[idx][idxcol_down_Qavg], df.iloc[idx][idxcol_down_Qstd]

		if int(df.iloc[idx]['parameter_2'])==22:
			color = color_ed22
		else:
			color = color_ed32

		ax0.plot((pupval_avg,pdownval_avg), (Qupval_avg,Qdownval_avg), color=color, marker = '',linewidth=3, alpha=.3,zorder=-10)

		ax0.errorbar(pupval_avg, Qupval_avg, xerr=pupval_std, yerr = Qupval_std, color=color, marker = '', capsize=capsize,zorder=-10)
		ax0.errorbar(pdownval_avg, Qdownval_avg, xerr=pdownval_std, yerr = Qdownval_std, color=color, marker = '',capsize=capsize,zorder=-10)
		ax0.scatter(pupval_avg, Qupval_avg, color='white', marker = '^',s=75, edgecolor=color,alpha=1)
		ax0.scatter(pdownval_avg, Qdownval_avg, color='white', marker = 'v', s=75, edgecolor=color,alpha=1)


		ax0.text(.5*(pupval_avg+pdownval_avg)-.15,.5*(Qupval_avg+Qdownval_avg),
		 (int(df.iloc[idx]['parameter_1'])),rotation=45,color=color)

ax0.set_xlabel(r"$p_{\mathrm{in}}$ [bar]", fontsize=ticks_size)
ax0.set_ylabel(r"$Q_{\mathrm{in}}$ [SLPM]", fontsize=ticks_size)

ax0.set_xlim(0.25,2.3)
ax0.set_ylim(50,190)
ax0.legend()


#ax1



for idx in range(len(df)):#

	if df.iloc[idx][idxcol_up_pavg] != 0 :

		pupval_avg, pupval_std = df.iloc[idx][idxcol_up_pavg], df.iloc[idx][idxcol_up_pstd]
		pdownval_avg, pdownval_std = df.iloc[idx][idxcol_down_pavg], df.iloc[idx][idxcol_down_pstd]

		Qupval_avg, Qupval_std = df.iloc[idx][idxcol_up_Qavg], df.iloc[idx][idxcol_up_Qstd]
		Qdownval_avg, Qdownval_std = df.iloc[idx][idxcol_down_Qavg], df.iloc[idx][idxcol_down_Qstd]

		if int(df.iloc[idx]['parameter_2'])==22:
			color = color_ed22
		else:
			color = color_ed32

		#ax0.scatter(pupval_avg, Qupval_avg, color=color, marker = '^',s=10*int(df.iloc[idx]['parameter_1']))
		ax1.scatter(int(df.iloc[idx]['parameter_1']),pupval_avg-pdownval_avg, color='white', marker = 'o', edgecolor=color,)
		ax2.scatter(int(df.iloc[idx]['parameter_1']),Qupval_avg-Qdownval_avg, color='white', marker = 'o', edgecolor=color,)

		#ax0.errorbar(pupval_avg, Qupval_avg, xerr=pupval_std, yerr = Qupval_std, color=color, marker = '', capsize=capsize )
		#ax0.errorbar(pdownval_avg, Qdownval_avg, xerr=pdownval_std, yerr = Qdownval_std, color=color, marker = '',capsize=capsize )
ax1.set_xlabel(r"Valve length [mm]", fontsize=ticks_size)
ax1.set_ylabel(r"$p_{\mathrm{in, forward}}-p_{\mathrm{in, backward}}$ [bar]", fontsize=ticks_size)
ax2.set_xlabel(r"Valve length [mm]", fontsize=ticks_size)
ax2.set_ylabel(r"$Q_{\mathrm{in, forward}}-Q_{\mathrm{in, backward}}$ [SLPM]", fontsize=ticks_size)

ax1.axvspan(12, 14, alpha=0.3, color='silver')
ax2.axvspan(12, 14, alpha=0.3, color='silver')
ax1.set_xlim(13,25)
ax2.set_xlim(13,25)


plt.tight_layout()
plt.savefig('qpthresholds.png', dpi=dpi, bbox_inches="tight")

plt.show()


fig = plt.figure(figsize=(12, 6))
ax0 = fig.add_subplot(132)
ax1 = fig.add_subplot(233)
ax2 = fig.add_subplot(236)
axes = [ax0,ax1,ax2]

for ax in axes:
    ax.xaxis.set_tick_params(labelsize=ticks_size)
    ax.yaxis.set_tick_params(labelsize=ticks_size)
    ax.grid(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
### ax0  




ax0.scatter(-10, -10, color='k', marker = '^', label = "Forward transition")
ax0.scatter(-10, -10, color='k', marker = 'v', label = "Backward transition")

ax0.scatter(-10, -10, color='white', edgecolor=color_ed22, marker = 'o', label = "ED22")
ax0.scatter(-10, -10, color='white',  edgecolor=color_ed32, marker = 'o', label = "ED32")

for idx in range(len(df)):#

	if df.iloc[idx][idxcol_up_pavg] != 0 :

		pupval_avg, pupval_std = df.iloc[idx][idxcol_up_pavg], df.iloc[idx][idxcol_up_pstd]
		pdownval_avg, pdownval_std = df.iloc[idx][idxcol_down_pavg], df.iloc[idx][idxcol_down_pstd]

		Qupval_avg, Qupval_std = df.iloc[idx][idxcol_up_Qavg], df.iloc[idx][idxcol_up_Qstd]
		Qdownval_avg, Qdownval_std = df.iloc[idx][idxcol_down_Qavg], df.iloc[idx][idxcol_down_Qstd]

		if int(df.iloc[idx]['parameter_2'])==22:
			color = color_ed22
		else:
			color = color_ed32

		ax0.plot( (Qupval_avg,Qdownval_avg),(pupval_avg,pdownval_avg), color=color, marker = '',linewidth=3, alpha=.3,zorder=-10)

		ax0.errorbar(Qupval_avg, pupval_avg,xerr=Qupval_std, yerr = pupval_std, color=color, marker = '', capsize=capsize,zorder=-10)
		ax0.errorbar(Qdownval_avg, pdownval_avg, xerr=Qdownval_std, yerr = pdownval_std, color=color, marker = '',capsize=capsize,zorder=-10)
		ax0.scatter(Qupval_avg, pupval_avg, color='white', marker = '^',s=75, edgecolor=color,alpha=1)
		ax0.scatter(Qdownval_avg, pdownval_avg, color='white', marker = 'v', s=75, edgecolor=color,alpha=1)


		ax0.text(.5*(Qupval_avg+Qdownval_avg)-15,.5*(pupval_avg+pdownval_avg),
		 (str(int(df.iloc[idx]['parameter_1']))),rotation=45,color=color)

ax0.set_ylabel(r"$p_{\mathrm{in}}$ [bar]", fontsize=ticks_size)
ax0.set_xlabel(r"$Q_{\mathrm{in}}$ [SLPM]", fontsize=ticks_size)


## Fluidic resistance in closed conditions line


prange_min, prange_max = np.sort(df['pin_down_avg'])[2],df['pin_up_avg'].max()
Qrange_min, Qrange_max = np.sort(df['Qin_down_avg'])[2],df['Qin_up_avg'].max()



unifiedQs = np.sort(np.concatenate([df['Qin_up_avg'].iloc[2:].to_numpy(),df['Qin_down_avg'].iloc[2:].to_numpy()]))
unifiedps = np.sort(np.concatenate([df['pin_up_avg'].iloc[2:].to_numpy(),df['pin_down_avg'].iloc[2:].to_numpy()]))

coeff = np.polyfit(unifiedQs, unifiedps, 2)
poly1d_fn = np.poly1d(coeff) 

print(coeff)
resistance_closedstate = coeff[0] #slope

ax0.plot(unifiedQs, poly1d_fn(unifiedQs), color='r', linestyle='--', label = '$R_{\\mathrm{fit,snap}}$')#=%.3f\\frac{\\mathrm{bar}}{\\mathrm{SLPM}}$'%(resistance_closedstate))


ax0.set_ylim(0.25,2.3)
ax0.set_xlim(50,190)
ax0.legend(fontsize=ticks_size)


#ax1



for idx in range(len(df)):#

	if df.iloc[idx][idxcol_up_pavg] != 0 :

		pupval_avg, pupval_std = df.iloc[idx][idxcol_up_pavg], df.iloc[idx][idxcol_up_pstd]
		pdownval_avg, pdownval_std = df.iloc[idx][idxcol_down_pavg], df.iloc[idx][idxcol_down_pstd]

		Qupval_avg, Qupval_std = df.iloc[idx][idxcol_up_Qavg], df.iloc[idx][idxcol_up_Qstd]
		Qdownval_avg, Qdownval_std = df.iloc[idx][idxcol_down_Qavg], df.iloc[idx][idxcol_down_Qstd]

		if int(df.iloc[idx]['parameter_2'])==22:
			color = color_ed22
		else:
			color = color_ed32

		#ax0.scatter(pupval_avg, Qupval_avg, color=color, marker = '^',s=10*int(df.iloc[idx]['parameter_1']))
		ax1.scatter(int(df.iloc[idx]['parameter_1']),pupval_avg-pdownval_avg, color='white', marker = 'o', edgecolor=color,)
		ax2.scatter(int(df.iloc[idx]['parameter_1']),Qupval_avg-Qdownval_avg, color='white', marker = 'o', edgecolor=color,)

		#ax0.errorbar(pupval_avg, Qupval_avg, xerr=pupval_std, yerr = Qupval_std, color=color, marker = '', capsize=capsize )
		#ax0.errorbar(pdownval_avg, Qdownval_avg, xerr=pdownval_std, yerr = Qdownval_std, color=color, marker = '',capsize=capsize )
ax1.axvspan(12, 14, alpha=0.3, color='silver')
ax2.axvspan(12, 14, alpha=0.3, color='silver')
ax1.set_xlim(13,25)
ax2.set_xlim(13,25)
ax1.set_xlabel(r"Valve length [mm]", fontsize=ticks_size)
ax1.set_ylabel(r"$p_{\mathrm{in, forward}}-p_{\mathrm{in, backward}}$ [bar]", fontsize=ticks_size)
ax2.set_xlabel(r"Valve length [mm]", fontsize=ticks_size)
ax2.set_ylabel(r"$Q_{\mathrm{in, forward}}-Q_{\mathrm{in, backward}}$ [SLPM]", fontsize=ticks_size)

#print(DeltaP, DeltaQ, DeltaP/DeltaQ)

plt.tight_layout()
plt.savefig('pqthresholds.png', dpi=dpi, bbox_inches="tight")

plt.show()