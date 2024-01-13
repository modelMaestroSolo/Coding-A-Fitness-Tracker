import pandas as pd
from glob import glob
from IPython.display import display


# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

single_file_gyr = pd.read_csv(
    "../../data/raw/MetaMotion/A-ohp-heavy_MetaWear_2019-01-14T14.49.46.484_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files = glob("../../data/raw/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
for i in range(187):
    file = files[i]
    split_list = file.split("-")
    participant = split_list[0][-1]
    label = split_list[1]
    category = split_list[2].rstrip("123").rstrip("_MetaWear_2019")
    print(participant, label, category)

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1


for filename in files:
    # extract features from file name
    split_list = filename.split("-")
    participant = split_list[0][-1]
    label = split_list[1]
    category = split_list[2].rstrip("123").rstrip("_MetaWear_2019")

    df = pd.read_csv(filename)  # read in data

    # append features to df
    df["participant"] = participant
    df["category"] = category
    df["label"] = label

    if "Accelerometer" in filename:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])

    if "Gyroscope" in filename:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]
# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")


def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for filename in files:
        # extract features from file name
        split_list = filename.split("-")
        participant = split_list[0][-1]
        label = split_list[1]
        category = split_list[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(filename)  # read in data

        # append features to df
        df["participant"] = participant
        df["category"] = category
        df["label"] = label

        if "Accelerometer" in filename:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

        if "Gyroscope" in filename:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)
# --------------------------------------------------------------
# Merging datasets

data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

# renaming columns of data_merged
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "category",
    "label",
    "set",
]

# --------------------------------------------------------------


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "category": "last",
    "label": "last",
    "set": "last",
}

# we settle for 200 ms period for the resampling.
# thus, gyroscope sample would have occured about 5 fives (0.2/0.04)
# and accelerometer samples aobut 2.5 times

# since data was collected over a week, applying resampling
# on entire merged df would mean that we interpolate values even
# for time periods when no data were collected
# we apply the resample according to days and recombine them


data_merged.resample(rule="200ms").apply(sampling)

# define a grouping rule using Grouper with day freq to
# group dfs into days. _ = the date of the days
days = [g for _, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

data_resampled.info()
data_resampled.set = data_resampled.set.astype("int")

##
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_preprocessed.pkl")
