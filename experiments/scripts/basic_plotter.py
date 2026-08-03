from pathlib import Path
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import cmcrameri.cm as cmc
import matplotlib.colors as colors

plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR / "experiment_data"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

color_left = "red"
color_right = "royalblue"

explinewidth = 0.5
simlinewidth = 4
alphasimline = 0.4
alphaexpline = 1
plotoffsetQ = 5
ticks_size = 14

# ---------------------------------------------------------------------
# Data reading
# ---------------------------------------------------------------------

def read_controller_csv(csv_path):
    """
    Recover only the 8-column controller rows from a mixed CSV file.

    Expected controller columns:
        repetition, controller_time, p_in,
        flow_0, flow_1, flow_2, p1, p2
    """

    controller_columns = [
        "repetition",
        "controller_time",
        "p_in",
        "flow_0",
        "flow_1",
        "flow_2",
        "p1",
        "p2",
    ]

    recovered_rows = []
    skipped_rows = 0

    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)

        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue

            # Ignore headers.
            if row[0].strip().lower() in {
                "repetition",
                "time_s",
            }:
                continue

            # Ignore DAQ rows and malformed rows.
            if len(row) != len(controller_columns):
                skipped_rows += 1
                continue

            try:
                values = [float(value) for value in row]
            except ValueError:
                print(f"Skipping invalid row at line {line_number}")
                skipped_rows += 1
                continue

            recovered_rows.append(values)

    if not recovered_rows:
        raise RuntimeError(
            "No valid 8-column controller rows were found in the CSV file."
        )

    df = pd.DataFrame(
        recovered_rows,
        columns=controller_columns,
    )

    df["repetition"] = df["repetition"].astype(int)

    df = (
        df
        .sort_values("controller_time")
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(f"Recovered {len(df)} controller rows.")
    print(f"Ignored {skipped_rows} DAQ or malformed rows.")
    print(
        "Available repetitions:",
        sorted(df["repetition"].unique()),
    )

    return df


# ---------------------------------------------------------------------
# Plot formatting
# ---------------------------------------------------------------------

def format_axes(axes):
    for ax in axes:
        ax.xaxis.set_tick_params(labelsize=ticks_size)
        ax.yaxis.set_tick_params(labelsize=ticks_size)
        ax.grid(False)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_basics(expfilename, timecutoff=0, plottitle = ''):
    csv_path = DATA_DIR / f"{expfilename}.csv"

    df = read_controller_csv(csv_path)

    # Keep only repetitions after repetition 0.
    #df = df[df["repetition"] == 1].copy()
    df = df[df["repetition"] == 1].copy()

    # Apply the optional time cutoff.
    df = df[df["controller_time"] > timecutoff].copy()

    if df.empty:
        raise RuntimeError(
            "No controller data remains after applying the repetition "
            "and time filters."
        )
    print(df.columns)
    # ---------------------------------------------------------------
    # Flow and pressure versus time
    # ---------------------------------------------------------------

    fig = plt.figure(figsize=(7, 7))
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    axes = [ax0, ax1]
    format_axes(axes)

    for ax in axes:
        ax.set_xlabel("Time [s]", fontsize=ticks_size)

    ax0.plot(
        df["controller_time"],
        df["flow_1"] * 200,
        color=color_left,
        label="$Q_\\mathrm{left}$",
    )

    ax0.plot(
        df["controller_time"],
        df["flow_2"] * 200,
        color=color_right,
        label="$Q_\\mathrm{right}$",
    )

    ax0.plot(
        df["controller_time"],
        df["flow_1"] * 200 + df["flow_2"] * 200,
        color='silver',
        label="$Q_\\mathrm{left}$+$Q_\\mathrm{right}$",
    )

    ax0.plot(
        df["controller_time"],
        df["flow_0"] * 200,
        color='k',
        label="$Q_\\mathrm{in}$",
    )
    ax0.legend(fontsize=ticks_size)
    ax1.plot(
        df["controller_time"],
        df["p_in"],
        color="k",
        label="$p_{\\mathrm{in}}$",
    )

    ax1.plot(
        df["controller_time"],
        df["p1"] * 7,
        color=color_right,
    )

    ax1.plot(
        df["controller_time"],
        df["p2"] * 7,
        color=color_left,
    )

    ax0.set_ylabel("$Q$ [SLPM]", fontsize=ticks_size)
    ax1.set_ylabel("$p$ [bar]", fontsize=ticks_size)

    plt.tight_layout()
    plotname = f"expdata_time_{expfilename}.png"

    plt.savefig(
        FIG_DIR / plotname,
        dpi=200,
    )

    plt.close(fig)

    # ---------------------------------------------------------------
    # Left versus right flow and pressure
    # ---------------------------------------------------------------

    fig = plt.figure(figsize=(8, 7))
    ax0 = fig.add_subplot(111)
    #ax1 = fig.add_subplot(212)
    ax0.set_title(plottitle,fontsize=ticks_size)

    axes = [ax0, ax1]
    format_axes(axes)

    ax0.set_ylabel(
        "$Q_{\\mathrm{left}}$ [SLPM]",
        fontsize=ticks_size,
    )

    ax0.set_xlabel(
        "$Q_{\\mathrm{right}}$ [SLPM]",
        fontsize=ticks_size,
    )
    cmap = cmc.managua #managua roma berlin_r

    cs = []

    #cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(df["flow_1"].min(), df["flow_1"].max())

    for i in range(len(df["flow_1"])):
        cs.append(cmap(i/len(df["flow_1"])))   
    print(df.columns)
    values = df["controller_time"]

    scatter = ax0.scatter(
    df["flow_1"] * 200,
    df["flow_2"] * 200,
    c=values,
    cmap=plt.get_cmap("plasma"),#cmc.managua,
    norm=colors.Normalize(values.min(), values.max()),
    alpha=0.7,
    marker="."
    )

    cbar = fig.colorbar(scatter, ax=ax0)
    cbar.set_label("Time [s]")
    ax0.set_xlim(0,100)
    ax0.set_ylim(0,100)
    minflow = 3#min(df["flow_1"].min(),df["flow_2"].min())
    maxflow = max(df["flow_1"].max(),df["flow_2"].max())
    for qtot in [30,50,70]:
        qtotx = np.linspace(minflow, qtot)
        qtoty = np.linspace(qtot,minflow)
        ax0.plot(qtotx,qtoty, color='silver', ls='--', linewidth=.5)
        ax0.text(np.mean(qtotx)-15,np.mean(qtoty)+10,'%d SLPM'%(qtot), color='k',rotation=-45)
    ax0.plot(np.linspace(minflow,100),np.linspace(minflow,100),color='k', ls='--')


    if False:
	    ax1.set_ylabel(
	        "$p_{\\mathrm{left}}$ [bar]",
	        fontsize=ticks_size,
	    )

	    ax1.set_xlabel(
	        "$p_{\\mathrm{right}}$ [bar]",
	        fontsize=ticks_size,
	    )

	    ax1.plot(
	        df["p1"] * 7,
	        df["p2"] * 7,
	        color="k",
	    )

    plt.tight_layout()

    plotname = f"qq{expfilename}.png"

    plt.savefig(
        FIG_DIR / plotname,
        dpi=200,
    )

    plt.close(fig)

    # ---------------------------------------------------------------
    # Flow versus pressure
    # ---------------------------------------------------------------

    fig = plt.figure(figsize=(7, 7))
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    axes = [ax0, ax1]
    format_axes(axes)

    for ax in axes:
        ax.set_ylabel("$p$ [bar]", fontsize=ticks_size)
        ax.set_xlabel("$Q$ [SLPM]", fontsize=ticks_size)

    ax0.plot(
        df["flow_1"] * 200, df["p1"] * 7,
        color=color_left,
    )

    ax1.plot(
        df["flow_2"] * 200, df["p2"] * 7,
        color=color_right,
    )

    plt.tight_layout()

    plotname = f"qp_{expfilename}.png"

    plt.savefig(
        FIG_DIR / plotname,
        dpi=200,
    )

    plt.close(fig)

    print(f"Figures saved in: {FIG_DIR}")

exps = []
expfilename = "20260802_172137" #60sbis
exps.append([expfilename,''])
expfilename = "20260802_162232" #60s
exps.append([expfilename,''])
expfilename = "20260802_162802" #250s
exps.append([expfilename,''])
expfilename = "20260802_161949" #30s
exps.append([expfilename,''])
expfilename = "20260803_151441" #30s
exps.append([expfilename,''])
expfilename = "20260803_172041" #10_20_7.5s_noglue
exps.append([expfilename,'10_20_7.5_noglue'])



for exp in exps:
	plot_basics(expfilename=exp[0],plottitle=exp[1])