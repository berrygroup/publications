"""
Utility functions for data loading and parameter management.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_parameters(filepath):
    """
    Load parameters from JSON file.

    Parameters
    ----------
    filepath : str or Path
        Path to JSON file

    Returns
    -------
    dict
        Parameter dictionary
    """
    with open(filepath, "r") as f:
        return json.load(f)


def save_parameters(params, filepath):
    """
    Save parameters to JSON file.

    Parameters
    ----------
    params : dict
        Parameter dictionary
    filepath : str or Path
        Path to save JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    params_serializable = {}
    for key, value in params.items():
        if isinstance(value, (np.integer, np.floating)):
            params_serializable[key] = float(value)
        else:
            params_serializable[key] = value

    with open(filepath, "w") as f:
        json.dump(params_serializable, f, indent=2)


def load_experimental_data(filepath, normalization_factors=None):
    """
    Load and normalize experimental data.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file with experimental data
    normalization_factors : dict, optional
        Dictionary with 'RFP' and 'GFP' normalization factors

    Returns
    -------
    pd.DataFrame
        Loaded and normalized data
    """
    data = pd.read_csv(filepath, index_col=0)

    if normalization_factors:
        if "RFP" in normalization_factors:
            data["RFP_Intensity_BleachCorrected"] *= normalization_factors["RFP"]
        if "GFP" in normalization_factors:
            data["GFP_Intensity_BleachCorrected"] *= normalization_factors["GFP"]

    return data


def calculate_time_averages(data, treatment_codes=None):
    """
    Calculate time-averaged experimental data for fitting.

    Parameters
    ----------
    data : pd.DataFrame
        Raw experimental data with columns: TREATMENT_CODE, TIME_HR,
        RFP_Intensity_BleachCorrected, GFP_Intensity_BleachCorrected
    treatment_codes : list, optional
        List of treatment codes to include. If None, use all.

    Returns
    -------
    dict
        Dictionary mapping treatment_code to DataFrame with averaged data
    """
    if treatment_codes is None:
        treatment_codes = data["TREATMENT_CODE"].unique()

    averaged_data = {}
    for code in treatment_codes:
        subset = data[data["TREATMENT_CODE"] == code]
        avg = (
            subset.groupby("TIME_HR")
            .agg(
                {
                    "RFP_Intensity_BleachCorrected": "mean",
                    "GFP_Intensity_BleachCorrected": "mean",
                }
            )
            .reset_index()
        )
        avg.columns = ["time", "rfp", "gfp"]
        averaged_data[code] = avg

    return averaged_data


def plot_data_preview(data_averaged, treatment_codes=None):
    """
    Create a quick preview plot of the averaged experimental data.

    Parameters
    ----------
    data_averaged : dict
        Dictionary mapping treatment_code to DataFrame with averaged data
    treatment_codes : list, optional
        List of treatment codes to plot. If None, plot all.
    """
    import matplotlib.pyplot as plt

    if treatment_codes is None:
        treatment_codes = list(data_averaged.keys())

    n_codes = len(treatment_codes)
    fig, axes = plt.subplots(1, n_codes, figsize=(6 * n_codes, 4), squeeze=False)
    axes = axes.flatten()

    colors = {"rfp": "#d62728", "gfp": "#2ca02c"}

    for ax, code in zip(axes, treatment_codes):
        data = data_averaged[code]

        # Plot data
        ax.scatter(
            data["time"],
            data["rfp"],
            color=colors["rfp"],
            alpha=0.6,
            s=60,
            label="RFP",
            zorder=3,
        )
        ax.scatter(
            data["time"],
            data["gfp"],
            color=colors["gfp"],
            alpha=0.6,
            s=60,
            marker="s",
            label="GFP",
            zorder=3,
        )

        ax.set_xlabel("Time (h)", fontsize=11)
        ax.set_ylabel("Nuclear concentration", fontsize=11)
        ax.set_title(code, fontsize=12)
        ax.legend(frameon=False)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
