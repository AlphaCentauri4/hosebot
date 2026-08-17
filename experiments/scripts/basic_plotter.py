from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors


plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


class ExperimentData:

    def __init__(
        self,
        expfilename,
        data_dir="experiment_data",
        figure_dir="figures",
        n_repetitions=1,
        repetition=0,
        timecutoff=0,
    ):
        self.expfilename = expfilename
        self.data_dir = Path(data_dir)
        self.figure_dir = Path(figure_dir)

        self.n_repetitions = n_repetitions
        self.repetition = repetition
        self.timecutoff = timecutoff

        self.figure_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.color_left = "red"
        self.color_right = "royalblue"

        self.ticks_size = 14

        self.csv_path = (
            self.data_dir
            / f"{self.expfilename}.csv"
        )

        self.df = self._read_controller_csv()

        self.data = self._get_repetition_data(
            self.repetition
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _read_controller_csv(self):

        controller_columns = [
            "time_s",
            "elapsed_us",
            "pres_in",
            "flow_in",
            "flow_left",
            "flow_right",
            "pres_left",
            "pres_right",
        ]

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_path}"
            )

        recovered_rows = []
        skipped_rows = 0

        with open(
            self.csv_path,
            "r",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            reader = csv.reader(csv_file)

            for line_number, row in enumerate(
                reader,
                start=1,
            ):

                if not row:
                    continue

                if row[0].strip().lower() in {
                    "time_s",
                    "time",
                }:
                    continue

                if len(row) != len(
                    controller_columns
                ):
                    skipped_rows += 1
                    continue

                try:
                    values = [
                        float(value)
                        for value in row
                    ]

                except ValueError:
                    print(
                        f"Skipping invalid row at line "
                        f"{line_number}: {row}"
                    )
                    skipped_rows += 1
                    continue

                recovered_rows.append(values)

        if not recovered_rows:
            raise RuntimeError(
                f"No valid controller rows were found "
                f"in {self.csv_path}."
            )

        df = pd.DataFrame(
            recovered_rows,
            columns=controller_columns,
        )

        df = (
            df
            .drop_duplicates()
            .reset_index(drop=True)
        )

        n_rows = len(df)

        rows_per_repetition = (
            n_rows // self.n_repetitions
        )

        if rows_per_repetition == 0:
            raise RuntimeError(
                f"Number of valid rows ({n_rows}) is too small "
                f"for {self.n_repetitions} repetitions "
                f"(need at least 1 row per repetition)."
            )

        n_rows_used = rows_per_repetition * self.n_repetitions
        leftover = n_rows - n_rows_used

        if leftover:
            print(
                f"Row count ({n_rows}) isn't evenly divisible by "
                f"{self.n_repetitions} repetitions; dropping the last "
                f"{leftover} row(s) so each repetition gets "
                f"{rows_per_repetition} rows."
            )

        df = df.iloc[:n_rows_used].reset_index(drop=True)

        df["repetition"] = np.repeat(
            np.arange(self.n_repetitions),
            rows_per_repetition,
        )

        df["controller_time"] = (
            df.groupby("repetition")["time_s"]
            .transform(
                lambda values:
                values - values.iloc[0]
            )
        )

        df["experiment_time"] = (
            df["time_s"] - df["time_s"].iloc[0]
        )

        print(
            f"Read {len(df)} controller rows."
        )

        print(
            f"Ignored {skipped_rows} malformed rows."
        )

        print(
            f"Detected {self.n_repetitions} "
            f"repetitions with "
            f"{rows_per_repetition} rows each."
        )

        return df

    def _get_repetition_data(
        self,
        repetition,
    ):

        if repetition < 0:
            raise ValueError(
                "Repetition must be non-negative."
            )

        if repetition >= self.n_repetitions:
            raise ValueError(
                f"Repetition {repetition} does not exist. "
                f"Valid repetitions are "
                f"0 through "
                f"{self.n_repetitions - 1}."
            )

        df = self.df[
            self.df["repetition"] == repetition
        ].copy()

        df = df[
            df["controller_time"]
            > self.timecutoff
        ].copy()

        if df.empty:
            raise RuntimeError(
                "No controller data remains after "
                f"applying repetition={repetition} "
                f"and timecutoff={self.timecutoff}."
            )

        return df

    def set_repetition(
        self,
        repetition,
    ):

        self.repetition = repetition

        self.data = self._get_repetition_data(
            repetition
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_axes(self, axes):

        for ax in axes:

            ax.xaxis.set_tick_params(
                labelsize=self.ticks_size
            )

            ax.yaxis.set_tick_params(
                labelsize=self.ticks_size
            )

            ax.grid(False)

            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(True)

    # ------------------------------------------------------------------
    # Time history
    # ------------------------------------------------------------------

    def plot_time_history(
        self,
        save=True,
        dpi=200,
        full_experiment=True,
    ):
        """
        By default this plots the whole experiment (every repetition,
        stitched together on a continuous time axis) rather than just
        the currently-selected repetition. Pass full_experiment=False
        to plot only self.data (the currently selected repetition) on
        its own repetition-relative time axis, as before.
        """

        if full_experiment:
            df = self.df[
                self.df["experiment_time"]
                > self.timecutoff
            ].copy()
            time_col = "experiment_time"
        else:
            df = self.data
            time_col = "controller_time"

        fig = plt.figure(
            figsize=(7, 7)
        )

        ax0 = fig.add_subplot(211)
        ax1 = fig.add_subplot(212)

        self.format_axes(
            [ax0, ax1]
        )

        for ax in [ax0, ax1]:

            ax.set_xlabel(
                "Time [s]",
                fontsize=self.ticks_size,
            )

        ax0.plot(
            df[time_col],
            df["flow_left"],
            color=self.color_left,
            label=r"$Q_{\mathrm{left}}$",
        )

        ax0.plot(
            df[time_col],
            df["flow_right"],
            color=self.color_right,
            label=r"$Q_{\mathrm{right}}$",
        )

        ax0.plot(
            df[time_col],
            (
                df["flow_left"]
                + df["flow_right"]
            ),
            color="silver",
            label=r"$Q_{\mathrm{left}} + Q_{\mathrm{right}}$",
        )

        ax0.plot(
            df[time_col],
            df["flow_in"],
            color="k",
            label=r"$Q_{\mathrm{in}}$",
        )

        ax0.set_ylabel(
            "$Q$ [SLPM]",
            fontsize=self.ticks_size,
        )

        ax0.legend(
            fontsize=self.ticks_size
        )

        ax1.plot(
            df[time_col],
            df["pres_in"],
            color="k",
            label=r"$p_{\mathrm{in}}$",
        )

        ax1.plot(
            df[time_col],
            df["pres_left"],
            color=self.color_left,
            label=r"$p_{\mathrm{left}}$",
        )

        ax1.plot(
            df[time_col],
            df["pres_right"],
            color=self.color_right,
            label=r"$p_{\mathrm{right}}$",
        )

        ax1.set_ylabel(
            "$p$ [bar]",
            fontsize=self.ticks_size,
        )

        ax1.legend(
            fontsize=self.ticks_size
        )

        if full_experiment and self.n_repetitions > 1:

            rep_starts = (
                df.groupby("repetition")[time_col]
                .min()
                .iloc[1:]
            )

            for ax in [ax0, ax1]:
                for rep_start in rep_starts:
                    ax.axvline(
                        rep_start,
                        color="silver",
                        linestyle="--",
                        linewidth=0.8,
                    )

        fig.tight_layout()

        if save:

            suffix = (
                "all"
                if full_experiment
                else f"rep{self.repetition}"
            )

            filename = (
                f"expdata_time_"
                f"{self.expfilename}_"
                f"{suffix}.png"
            )

            path = (
                self.figure_dir
                / filename
            )

            fig.savefig(
                path,
                dpi=dpi,
                bbox_inches="tight",
            )

            print(
                f"Saved figure: {path}"
            )

        return fig

    # ------------------------------------------------------------------
    # Flow versus flow
    # ------------------------------------------------------------------

    def plot_flow_flow(
        self,
        title="",
        save=True,
        dpi=200,
    ):

        df = self.data

        fig = plt.figure(
            figsize=(8, 7)
        )

        ax = fig.add_subplot(111)

        ax.set_title(
            title,
            fontsize=self.ticks_size,
        )

        self.format_axes([ax])

        ax.set_ylabel(
            r"$Q_{\mathrm{left}}$ [SLPM]",
            fontsize=self.ticks_size,
        )

        ax.set_xlabel(
            r"$Q_{\mathrm{right}}$ [SLPM]",
            fontsize=self.ticks_size,
        )

        values = df["controller_time"]

        scatter = ax.scatter(
            df["flow_left"],
            df["flow_right"],
            c=values,
            cmap=plt.get_cmap("plasma"),
            norm=colors.Normalize(
                values.min(),
                values.max(),
            ),
            alpha=0.7,
            marker=".",
        )

        cbar = fig.colorbar(
            scatter,
            ax=ax,
        )

        cbar.set_label(
            "Time [s]",
            fontsize=self.ticks_size,
        )

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        minflow = 3

        for qtot in [30, 50, 70]:

            qtotx = np.linspace(
                minflow,
                qtot,
            )

            qtoty = np.linspace(
                qtot,
                minflow,
            )

            ax.plot(
                qtotx,
                qtoty,
                color="silver",
                linestyle="--",
                linewidth=0.5,
            )

            ax.text(
                np.mean(qtotx) - 15,
                np.mean(qtoty) + 10,
                f"{qtot:d} SLPM",
                color="k",
                rotation=-45,
            )

        ax.plot(
            np.linspace(
                minflow,
                100,
            ),
            np.linspace(
                minflow,
                100,
            ),
            color="k",
            linestyle="--",
        )

        fig.tight_layout()

        if save:

            filename = (
                f"qq_"
                f"{self.expfilename}_"
                f"rep{self.repetition}.png"
            )

            path = (
                self.figure_dir
                / filename
            )

            fig.savefig(
                path,
                dpi=dpi,
                bbox_inches="tight",
            )

            print(
                f"Saved figure: {path}"
            )

        return fig

    # ------------------------------------------------------------------
    # Flow versus pressure
    # ------------------------------------------------------------------

    def plot_flow_pressure(
        self,
        save=True,
        dpi=200,
    ):

        df = self.data

        fig = plt.figure(
            figsize=(7, 7)
        )

        ax0 = fig.add_subplot(211)
        ax1 = fig.add_subplot(212)

        self.format_axes(
            [ax0, ax1]
        )

        for ax in [ax0, ax1]:

            ax.set_ylabel(
                "$p$ [bar]",
                fontsize=self.ticks_size,
            )

            ax.set_xlabel(
                "$Q$ [SLPM]",
                fontsize=self.ticks_size,
            )

        ax0.plot(
            df["flow_left"],
            df["pres_left"],
            color=self.color_left,
        )

        ax1.plot(
            df["flow_right"],
            df["pres_right"],
            color=self.color_right,
        )

        fig.tight_layout()

        if save:

            filename = (
                f"qp_"
                f"{self.expfilename}_"
                f"rep{self.repetition}.png"
            )

            path = (
                self.figure_dir
                / filename
            )

            fig.savefig(
                path,
                dpi=dpi,
                bbox_inches="tight",
            )

            print(
                f"Saved figure: {path}"
            )

        return fig

    # ------------------------------------------------------------------
    # All plots
    # ------------------------------------------------------------------

    def plot_all(
        self,
        title="",
        save=True,
        show=True,
        dpi=200,
        full_experiment=True,
    ):

        figures = []

        figures.append(
            self.plot_time_history(
                save=save,
                dpi=dpi,
                full_experiment=full_experiment,
            )
        )

        figures.append(
            self.plot_flow_flow(
                title=title,
                save=save,
                dpi=dpi,
            )
        )

        figures.append(
            self.plot_flow_pressure(
                save=save,
                dpi=dpi,
            )
        )

        if save:
            print(
                f"All figures saved in: "
                f"{self.figure_dir}"
            )

        if show:
            plt.show(block=False)
            plt.pause(0.1)
            input("Press Enter to close plots and continue...")
            plt.close("all")
            print("Plots closed.")

        return figures


if __name__ == "__main__":

    experiment = ExperimentData(
        expfilename="20260811_173910",
        data_dir="experiment_data",
        figure_dir="figures",
        n_repetitions=1,
        repetition=0,
    )

    experiment.plot_all(
        title="",
        save=True,
        show=True,
    )