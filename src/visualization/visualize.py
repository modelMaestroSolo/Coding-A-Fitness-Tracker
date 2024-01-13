import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Goal: to create data visualizations to better understand
# accelerometer and gyroscope data for the different exercises

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_preprocessed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set_df = df.query("set == 1")
plt.plot(set_df.reset_index().acc_y)  # reset index to plot num of samples xaxis
set_df.reset_index()

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()

    # choose first 100 samples to reduce clutter and see clear differences

    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)  # good size for reports
mpl.rcParams["figure.dpi"] = 100  # good looking resolutions in reports


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
# indeed heavy sets are expected to be done with relatively
# smaller vertical acceleration.

category_df = df.query("label == 'squat' and participant == 'A'").reset_index()

fig, ax = plt.subplots()
category_df.groupby("category")["acc_y"].plot()
ax.set_xlabel("samples")
ax.set_ylabel("acc_y")
ax.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

# sort_values() necessary to ensure that
# participant acc_y for both medium and heavy occur together in the indices.
# experiment by removing .sort_values to see the output visualization.
# groupby ensure same color, sort_values ensures same location along x-axis

participant_df = df.query("label=='bench'").sort_values("participant").reset_index()
subset = participant_df.groupby("participant")["acc_y"]
subset.plot()

# showing fair similarities accross participants. generalization to
# new participants should be feasible

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label = "squat"
participant = "A"
all_axis_df = df.query(
    f"label=='{label}' and participant=='{participant}'"
).reset_index()

fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_xlabel("samples")
ax.set_ylabel("acc")
ax.legend()


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
for label in df.label.unique()[:4]:
    for participant in df.participant.unique():
        all_axis_df = df.query(
            f"label=='{label}' and participant=='{participant}'"
        ).reset_index()
        if len(all_axis_df) != 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_xlabel("samples")
            ax.set_title(f"{participant}/{label}")
            ax.legend()


# for gyroscope
for label in df.label.unique()[:4]:
    for participant in df.participant.unique():
        all_axis_df = df.query(
            f"label=='{label}' and participant=='{participant}'"
        ).reset_index()
        if len(all_axis_df) != 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_xlabel("samples")
            ax.set_title(f"{participant}/{label}")
            ax.legend()


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
for label in df.label.unique():
    for participant in df.participant.unique():
        all_axis_df = df.query(
            f"label=='{label}' and participant=='{participant}'"
        ).reset_index()

        if len(all_axis_df) != 0:
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax1)
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax2)

            ax1.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax2.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )

            ax1.set_xlabel("Samples")
            fig.suptitle(f"{label} -- {participant}".title(), fontsize=18)
            plt.legend()

            plt.savefig(f"../../reports/figures/{label.title()}({participant}).png")
            plt.show()

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
