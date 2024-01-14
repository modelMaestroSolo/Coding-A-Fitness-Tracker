import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_preprocessed.pkl")
outlier_columns = list(df.columns[:6])

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

# boxplot shows quite a huge number of outliers.
# we can view the exact data points to confirm if these are indeed outliers
# or if they're pretty normal and the outlier technique used here isn't really
# the best for this given data set

df[outlier_columns[:3] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1, 3))
df[outlier_columns[3:] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1, 3))
plt.show()

# the boxplot doesn't show use us the locations of outliers for us to see based
# on how data is distributed in time whether they're outliers indeed
# when data is plotted, we may see clusters which may be an indication of
# heterogeneity of data, causing non-outliers to appear as outliers
# when points are plotted in time, check out for clusters, and where indeed
# outliers are far from the rest of the data.
# or an entire bulk of data (or clusters are all flagged as outliers)

# use custom function to visualize outliers in time:
# adjusted from the official source to fit Dave's style


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

# Insert IQR function


def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


# Plot a single column
col = "acc_x"
dataset = mark_outliers_iqr(df, col=col)
plot_binary_outliers(
    dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
)

# we see bulk of data (far right) flagged as outliers,
# plus some points are too close (middle bulk)  to consider as outliers.
# some are fairly understandable.


# Loop over all columns
for col in outlier_columns:
    dataset = mark_outliers_iqr(df, col=col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

# gyr data has more outliers compared to acc data. deleting all of them can
# significantly impact our dataset size.
# since iqr is a distribution based, if there are clusters, whose statistical
# properties such as the mean are sig different
# the larger group will dominate the small one, and make it appear as if
# values in the smaller group are outliers
# it is clearly better to separate the data before we apply these outlier detection
# method.
# we discuss other two outlier detection method then split data according to
# label before detecting outliers

# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution
df[outlier_columns[:3] + ["label"]].plot.hist(
    by="label", figsize=(20, 20), layout=(3, 3)
)

df[outlier_columns[3:] + ["label"]].plot.hist(
    by="label", figsize=(20, 20), layout=(3, 3)
)
plt.show()

# hence it'll make much sense to rather check normality on the columns
# and not the columns separated into groups by label


# personal observation: the outlier detection is performed on the
# columns(not separated into groups by label )

df[outlier_columns[:3]].plot.hist(figsize=(20, 10), layout=(1, 3), subplots=True)

df[outlier_columns[3:]].plot.hist(figsize=(20, 10), layout=(1, 3), subplots=True)
plt.show()


# the acc data is fairly normal in some cases, and not normal in some cases
# such as rest and dead(if we consider)

# Insert Chauvenet's function


def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.

    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


# Loop over all columns
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df, col=col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

# this method is not soo hard on the data,
# leaves a lot of data untagged as outliers

# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

# Insert LOF function


def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


# Loop over all columns
dataset, outliers, X_scores = mark_outliers_lof(df, outlier_columns)

for col in outlier_columns:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )

# it makes sense that the outliers occur at the same places
# for each of the plots as this is row based technique

# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------
label = "bench"
for col in outlier_columns:
    dataset = mark_outliers_iqr(df[df["label"] == label], col=col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )
# with even improving the homogeneit of the data by grouping using
# label a large chunk of data is still flagged as outliers with iqr
# method. experiment by changing label value to say bench

label = "squat"
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df[df["label"] == label], col=col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )
# chauvenet method is faily lenient in flagging data points as outliers
# optimal choice if grouped data passed normality test.

label = "squat"
dataset, outliers, X_scores = mark_outliers_lof(
    df[df["label"] == label], columns=outlier_columns
)
for col in outlier_columns:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )

# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------


# Test on single column
label = "bench"
col = "gyr_z"
dataset = mark_outliers_chauvenet(df[df["label"] == label], col=col)
dataset.loc[dataset["gyr_z_outlier"], col] = np.nan  # replace outlier with na
dataset.loc[dataset["gyr_z_outlier"]]

# Create a loop
outliers_removed_df = df.copy()

for col in outlier_columns:
    for label in df.label.unique():
        dataset = mark_outliers_chauvenet(df[df["label"] == label], col=col)
        dataset.loc[dataset[col + "_outlier"], col] = np.nan

        outliers_removed_df.loc[(outliers_removed_df["label"] == label), col] = dataset[
            col
        ]

        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f"Removed {n_outliers} from {col} for {label}")

outliers_removed_df.info()

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
outliers_removed_df.to_pickle("../../data/interim/02_removed_chauvenets.pkl")
# outlier removal may require domain knowledge to validate whether
# a particular approach is valid or not
