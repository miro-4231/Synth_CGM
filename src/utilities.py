import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay
)
import gc
import random 
import logging
import datetime
from .models import StandardModel
# from liner import
import seaborn as sns 
import torch 
import shap
from scipy.signal import correlate

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
log_filename = f"d:\\Projects\\PythonProjects\\PFE_CAD\\CGM_PROJECT\\Logs\\tuning_{timestamp}.log"

# Configure Logger
logging.basicConfig(
    filename=log_filename,
    filemode="w",  # Overwrite on each run
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


def clarke_error_grid_zones(y_true, y_pred):
    """
    Determines the Clarke Error Grid zone for each pair of true and predicted glucose values.
    Zone Definitions (common, but can vary slightly in literature):
    A: Clinically accurate
    B: Benign errors (acceptable)
    C: Over-correction errors (unnecessary treatment)
    D: Failure to detect errors (dangerous)
    E: Erroneous treatment errors (very dangerous)

    Returns a list of zone labels ('A', 'B', 'C', 'D', 'E') for each data point.
    """
    # Ensure inputs are 1D NumPy arrays
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_flat = np.asarray(y_pred).flatten()

    if len(y_true_flat) != len(y_pred_flat):
        raise ValueError("y_true and y_pred must have the same number of elements.")

    zones = []
    for i in range(len(y_true_flat)): # Iterate using an index
        true_val = y_true_flat[i]   # Get scalar value
        pred_val = y_pred_flat[i]   # Get scalar value
        
        zone = '' # Initialize zone for this point

        # Zone E: Erroneous treatment (most critical, check first)
        if (true_val <= 70 and pred_val >= 180) or \
           (true_val >= 180 and pred_val <= 70):
            zone = 'E'
        # Zone D: Failure to detect
        elif not zone and \
             ((true_val <= 70 and (pred_val > 70 and pred_val < 180)) or \
              (true_val >= 240 and (pred_val > 70 and pred_val < 180)) #or \
            #   (pred_val <= 70 and (true_val > 70 and true_val < 180)) or \
            #   (pred_val >= 180 and (true_val > 70 and true_val < 180))
            ):
            zone = 'D'
        # Zone C: Over-correction
        elif not zone and \
             ((true_val > 70 and true_val < 250 and (pred_val > 180 and pred_val > true_val + 110)) or # Predicted much higher
              (true_val > 70 and true_val < 180 and (pred_val < 70 and pred_val < true_val - 130)) # or # Predicted much lower
              # Symmetrical C for completeness, though less common in some definitions
            #   (pred_val > 70 and pred_val < 180 and (true_val > 180 and true_val > pred_val + 60)) or
            #   (pred_val > 70 and pred_val < 180 and (true_val < 70 and true_val < pred_val - 40))
              ):
            zone = 'C' # More specific C definitions might be needed
        # Zone A: Clinically accurate
        elif not zone and \
             ((true_val >= 70 and pred_val >=70 and abs(true_val - pred_val) <= 0.2 * true_val) or \
              (true_val < 70 and pred_val < 70)):# and abs(true_val - pred_val) <= 15) or \
            #   (true_val < 70 and abs(true_val - pred_val) <= 15 ) or \
            #   (pred_val < 70 and abs(true_val - pred_val) <= 15 ) ) : # inclusive A for low values if one is low
            zone = 'A'
        # Zone B: Benign errors (everything else)
        elif not zone:
            zone = 'B'
        
        # # Final check for A if it fell into B but meets strict A
        # if zone == 'B':
        #      if (true_val >= 70 and pred_val >=70 and abs(true_val - pred_val) <= 0.2 * true_val) or \
        #         (true_val < 70 and pred_val < 70 and abs(true_val - pred_val) <= 15) :
        #          zone = 'A'
                 
        zones.append(zone)
    return zones


def plot_clarke_error_grid(y_true, y_pred, title='Clarke Error Grid'):
    """
    Plots the Clarke Error Grid.
    Assumes y_true and y_pred are 1D arrays of glucose values.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, c='black', s=8, alpha=0.3) # Plot data points

    # Ideal line (y=x)
    plt.plot([0, 400], [0, 400], 'k:', lw=1) # Adjusted range for typical glucose values

    # Zone lines (common approximations)
    # These lines are approximations for visualization.
    # For exact zone percentages, use the zone calculation function.
    
    # A-B boundary (approx. 20% or 15mg/dL for low values)
    plt.plot([0, 400], [0, 400*0.8], 'g-', lw=1)
    plt.plot([0, 400], [0, 400*1.2], 'g-', lw=1)
    plt.plot([0, 70], [0+15, 70+15], 'g-', lw=1, alpha=0.5)
    plt.plot([0, 70-15], [0+15-15, 70-15+15], 'g-', lw=1, alpha=0.5)


    # Critical hypoglycemia/hyperglycemia lines
    plt.axhline(y=70, color='r', linestyle='--', lw=1, alpha=0.7)
    plt.axvline(x=70, color='r', linestyle='--', lw=1, alpha=0.7)
    plt.axhline(y=180, color='orange', linestyle='--', lw=1, alpha=0.7)
    plt.axvline(x=180, color='orange', linestyle='--', lw=1, alpha=0.7)
    
    # Fill regions (This is a simplified visual representation)
    # For accurate zone counting, use the clarke_error_grid_zones function
    plt.fill_between([0, 400], [0, 400*0.8], [0,0], color='green', alpha=0.1, label='Zone A/B (approx)')
    plt.fill_between([0, 400], [0, 400*1.2], [400,400], color='green', alpha=0.1)

    # Zone E approx regions
    plt.fill_between([0, 70], [180, 180], [400,400], color='red', alpha=0.2, label='Zone E (approx)')
    plt.fill_between([180, 400], [0,0], [70,70], color='red', alpha=0.2)


    plt.xlabel('Actual Glucose (mg/dL)')
    plt.ylabel('Predicted Glucose (mg/dL)')
    plt.title(title)
    plt.xlim(0, 400)
    plt.ylim(0, 400)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper left', fontsize='small')
    plt.show()

    # Calculate and print zone percentages
    zones = clarke_error_grid_zones(y_true, y_pred)
    zone_counts = {zone: zones.count(zone) for zone in ['A', 'B', 'C', 'D', 'E']}
    total_points = len(zones)
    zone_percentages = {zone: (count / total_points) * 100 for zone, count in zone_counts.items()}
    
    print("Clarke Error Grid Zone Percentages:")
    for zone, percentage in zone_percentages.items():
        print(f"  Zone {zone}: {percentage:.2f}% ({zone_counts[zone]} points)")
    
    return zone_percentages # Return percentages for storage

def calculate_time_gain_robust(actual_glucose_horizon,
                               predicted_glucose_horizon,
                               prediction_horizon_steps,
                               time_step_minutes=5):
    """
    Calculate the Time Gain (TG) robustly using cross-correlation.
    This version assumes inputs are the actual and predicted values *for the prediction horizon window*.

    Parameters:
    -----------
    actual_glucose_horizon : numpy.ndarray
        Array of actual glucose readings *over the prediction horizon* (1D).
        Example: If PH=6 steps, this should be 6 true future values.
    predicted_glucose_horizon : numpy.ndarray
        Array of predicted glucose readings *for the same prediction horizon* (1D).
        Example: If PH=6 steps, this should be the 6 predicted values.
    prediction_horizon_steps : int
        The length of the prediction horizon in number of time steps (e.g., 6 for a 30-min PH with 5-min steps).
        This should match the length of actual_glucose_horizon and predicted_glucose_horizon.
    time_step_minutes : int, optional
        The duration of one time step in minutes (default is 5).

    Returns:
    --------
    time_gain_minutes : float
        The calculated time gain in minutes. Returns np.nan if calculation is not possible.
    delay_minutes : float
        The calculated delay in minutes.
        Positive delay: Prediction lags actual (predicted events occur later than actual).
        Negative delay: Prediction leads actual.
        Returns np.nan if calculation is not possible.
    """
    actual = np.asarray(actual_glucose_horizon).flatten()
    predicted = np.asarray(predicted_glucose_horizon).flatten()

    if len(actual) != prediction_horizon_steps or len(predicted) != prediction_horizon_steps:
        logger.warning(
            f"TimeGain: Input array lengths ({len(actual)}, {len(predicted)}) "
            f"do not match prediction_horizon_steps ({prediction_horizon_steps}). "
            "TG calculation might be misleading. Ensure inputs are sliced correctly for the horizon."
        )
        # Attempt to proceed if lengths are at least > 1, otherwise, it's impossible.
        # This case should ideally be handled by the caller ensuring correct slicing.
        if len(actual) < 2 or len(predicted) < 2:
            return np.nan, np.nan
        # If lengths are different but >1, take the minimum for correlation
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]


    if len(actual) < 2 : # Cross-correlation needs at least 2 points
        logger.warning("TimeGain: Not enough data points for cross-correlation. Returning NaN.")
        return np.nan, np.nan

    # Normalize signals: (signal - mean) / std. Add epsilon for numerical stability.
    actual_norm = (actual - np.mean(actual)) / (np.std(actual) + 1e-9)
    predicted_norm = (predicted - np.mean(predicted)) / (np.std(predicted) + 1e-9)

    # Handle cases where std might be zero (e.g., flat line) leading to NaNs after normalization
    if np.isnan(actual_norm).any() or np.isnan(predicted_norm).any():
        logger.warning("TimeGain: NaN encountered after normalization (likely due to zero std dev). Returning NaN.")
        return np.nan, np.nan

    # Calculate cross-correlation
    # 'full' mode gives all possible overlaps, 'valid' only where they fully overlap,
    # 'same' returns output of same size as max(len(actual_norm), len(predicted_norm)) centered.
    correlation = correlate(actual_norm, predicted_norm, mode='full')
    
    # Lags corresponding to the 'full' correlation
    # The peak of the correlation indicates the shift of `predicted_norm` relative to `actual_norm`
    lags = np.arange(-(len(predicted_norm) - 1), len(actual_norm))
    
    # delay_steps is the shift that aligns predicted with actual.
    # A positive delay_steps means `predicted` is shifted to the right (lags `actual`).
    # A negative delay_steps means `predicted` is shifted to the left (leads `actual`).
    delay_steps = lags[np.argmax(correlation)]

    # Time Gain Calculation:
    # TG = PH_effective - delay_to_align_prediction_to_actual
    # PH_effective is the intended lead time (prediction_horizon_steps)
    # If prediction is perfect and aligned, delay_steps = 0, TG = PH.
    # If prediction lags by `delay_steps`, the actual warning time is PH - delay_steps.
    time_gain_steps = prediction_horizon_steps - delay_steps
    
    time_gain_minutes = time_gain_steps * time_step_minutes
    delay_minutes = delay_steps * time_step_minutes

    return time_gain_minutes, delay_minutes

# def regression_report(y_true, y_pred, plot=True):
#     """
#     Generates a comprehensive regression report with common metrics and an optional plot.

#     Parameters:
#     - y_true (array-like): True target values.
#     - y_pred (array-like): Predicted target values.
#     - plot (bool): Whether to plot actual vs. predicted values. Default is True.

#     Returns:
#     - report (dict): A dictionary containing regression metrics.
#     """

#     y_pred = y_pred.reshape(-1, 1)
#     y_true = y_true.reshape(-1, 1)
#     # Calculate regression metrics
#     mae = mean_absolute_error(y_true, y_pred)
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_true, y_pred)
#     mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
#     #time_gain = calculate_time_gain(y_true, y_pred, ph)

#     # Calculate correlation coefficients
#     #pearson_corr, _ = pearsonr(y_true, y_pred)
#     #spearman_corr, _ = spearmanr(y_true, y_pred)

#     # Create the report
#     report = {
#         "Mean Absolute Error (MAE)": mae,
#         "Mean Squared Error (MSE)": mse,
#         "Root Mean Squared Error (RMSE)": rmse,
#         "Mean Absolute Percentage Error (MAPE)": mape,
#         "R^2 Score": r2,
#        # "Time Gain": time_gain
#     }

#     # Optionally plot actual vs. predicted values
#     if plot:
#         plt.figure(figsize=(10, 6))
#         plt.scatter(y_true, y_pred, alpha=0.7, color='blue', label='Predictions')
#         plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Ideal')
#         plt.title('Actual vs. Predicted Values')
#         plt.xlabel('Actual Values')
#         plt.ylabel('Predicted Values')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#     return report

# def regression_report_gain(y_true_sequences, y_pred_sequences, plot=True,
#                       prediction_horizon_steps=None, time_step_minutes=5, delay:bool = False):
#     """
#     Generates a comprehensive regression report.
#     Time Gain is calculated per sequence and then averaged if y_true/y_pred are 2D.

#     Parameters:
#     - y_true_sequences (array-like): True target values. Can be 1D (single sequence) or 2D (multiple sequences, n_samples x PH).
#     - y_pred_sequences (array-like): Predicted target values. Shape must match y_true_sequences.
#     - prediction_horizon_steps (int, optional): The prediction horizon in number of time steps.
#                                                  If y_true/y_pred are 2D, this should be y_true_sequences.shape[1].
#     - time_step_minutes (int, optional): Duration of one time step in minutes.
#     """
#     y_true = np.asarray(y_true_sequences)
#     y_pred = np.asarray(y_pred_sequences)

#     if y_true.shape != y_pred.shape:
#         logger.error("RegressionReport: y_true and y_pred shapes mismatch. Cannot proceed.")
#         return {"Error": "Shape mismatch between y_true and y_pred."}

#     # Calculate overall metrics on flattened data
#     y_true_flat = y_true.flatten()
#     y_pred_flat = y_pred.flatten()

#     mae = mean_absolute_error(y_true_flat, y_pred_flat)
#     mse = mean_squared_error(y_true_flat, y_pred_flat)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_true_flat, y_pred_flat)
#     mask = y_true_flat != 0
#     mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100 if np.any(mask) else np.nan

#     report = {
#         "Mean Absolute Error (MAE)": mae,
#         "Mean Squared Error (MSE)": mse,
#         "Root Mean Squared Error (RMSE)": rmse,
#         "Mean Absolute Percentage Error (MAPE)": mape,
#         "R^2 Score": r2,
#     }

#     if prediction_horizon_steps is not None:
#         if y_true.ndim == 1: # Single sequence
#             tg_val, delay_val = calculate_time_gain_robust(
#                 y_true, y_pred, prediction_horizon_steps, time_step_minutes
#             )
#             report["Time Gain (minutes)"] = tg_val
#             report["Delay (minutes)"] = delay_val
#         elif y_true.ndim == 2: # Multiple sequences (n_samples, PH)
#             if y_true.shape[1] != prediction_horizon_steps:
#                 logger.warning(
#                     f"RegressionReport TG: y_true.shape[1] ({y_true.shape[1]}) "
#                     f"does not match prediction_horizon_steps ({prediction_horizon_steps}). "
#                     "Using y_true.shape[1] for PH in TG calculation."
#                 )
#                 current_ph_steps = y_true.shape[1]
#             else:
#                 current_ph_steps = prediction_horizon_steps

#             time_gains = []
#             delays = []
#             for i in range(y_true.shape[0]):
#                 tg_seq, delay_seq = calculate_time_gain_robust(
#                     y_true[i, :], y_pred[i, :], current_ph_steps, time_step_minutes
#                 )
#                 if not np.isnan(tg_seq):
#                     time_gains.append(tg_seq)
#                 if not np.isnan(delay_seq):
#                     delays.append(delay_seq)
            
#             report["Average Time Gain (minutes)"] = np.mean(time_gains) if time_gains else np.nan
#             report["Std Time Gain (minutes)"] = np.std(time_gains) if time_gains else np.nan
#             report["Average Delay (minutes)"] = np.mean(delays) if delays else np.nan
#             report["Std Delay (minutes)"] = np.std(delays) if delays else np.nan
#         else:
#             report["Time Gain (minutes)"] = np.nan
#             report["Delay (minutes)"] = np.nan
#             logger.warning("TimeGain: y_true/y_pred have unsupported dimensions for TG calculation.")

#     if plot:
#         # Plotting logic remains the same, using flattened versions
#         plt.figure(figsize=(10, 6))
#         # ... (rest of plotting code using y_true_flat, y_pred_flat)
#         plt.show()
        
#     if delay: 
#         return report, delays

#     return report

def regression_report(y_true_sequences, y_pred_sequences, plot_scatter=True, plot_ceg=True,
                      prediction_horizon_steps=None, time_step_minutes=5,
                      return_individual_delays=False): # Added plot_ceg flag
    """
    Generates a comprehensive regression report with common metrics and optional plots.
    Includes robust time gain calculation and Clarke Error Grid.

    Parameters:
    - y_true_sequences (array-like): True target values. Can be 1D or 2D (n_samples x PH).
    - y_pred_sequences (array-like): Predicted target values. Shape must match.
    - plot_scatter (bool): Whether to plot actual vs. predicted scatter.
    - plot_ceg (bool): Whether to plot the Clarke Error Grid.
    - prediction_horizon_steps (int, optional): PH in steps for Time Gain.
    - time_step_minutes (int, optional): Duration of one time step in minutes.
    - return_individual_delays (bool): If true, also returns list of individual delays.

    Returns:
    - report (dict): Dictionary containing regression metrics and CEG percentages.
    - (optional) all_delays_for_sequences (list): List of individual delays if return_individual_delays is True.
    """
    y_true = np.asarray(y_true_sequences)
    y_pred = np.asarray(y_pred_sequences)

    if y_true.shape != y_pred.shape:
        logger.error("RegressionReport: y_true and y_pred shapes mismatch. Cannot proceed.")
        return {"Error": "Shape mismatch between y_true and y_pred."}

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # --- Standard Regression Metrics (MAE, MSE, RMSE, R2, MAPE) ---
    # ... (same as your existing code for these metrics) ...
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_flat, y_pred_flat)
    mask = y_true_flat != 0
    mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100 if np.any(mask) else np.nan

    report = {
        "Mean Absolute Error (MAE)": mae,
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": rmse,
        "Mean Absolute Percentage Error (MAPE)": mape,
        "R^2 Score": r2,
    }

    # --- Time Gain Calculation ---
    all_delays_for_sequences = [] # Initialize for optional return
    if prediction_horizon_steps is not None:
        # ... (your existing robust time gain calculation logic calling calculate_time_gain_robust) ...
        # ... (ensure it populates 'report' with Avg Time Gain, Std Time Gain, Avg Delay, Std Delay) ...
        # ... and `all_delays_for_sequences` if needed for `return_individual_delays`
        if y_true.ndim == 1: 
            tg_val, delay_val = calculate_time_gain_robust(
                y_true, y_pred, prediction_horizon_steps, time_step_minutes
            )
            report["Time Gain (minutes)"] = tg_val
            report["Delay (minutes)"] = delay_val
            if not np.isnan(delay_val): all_delays_for_sequences.append(delay_val)

        elif y_true.ndim == 2:
            if y_true.shape[1] != prediction_horizon_steps:
                logger.warning(
                    f"RegressionReport TG: y_true.shape[1] ({y_true.shape[1]}) "
                    f"does not match prediction_horizon_steps ({prediction_horizon_steps}). "
                    "Using y_true.shape[1] for PH in TG calculation."
                )
                current_ph_steps = y_true.shape[1]
            else:
                current_ph_steps = prediction_horizon_steps

            time_gains_indiv = []
            delays_indiv = []
            for i in range(y_true.shape[0]):
                tg_seq, delay_seq = calculate_time_gain_robust(
                    y_true[i, :], y_pred[i, :], current_ph_steps, time_step_minutes
                )
                if not np.isnan(tg_seq):
                    time_gains_indiv.append(tg_seq)
                if not np.isnan(delay_seq):
                    delays_indiv.append(delay_seq)
            
            report["Average Time Gain (minutes)"] = np.mean(time_gains_indiv) if time_gains_indiv else np.nan
            report["Std Time Gain (minutes)"] = np.std(time_gains_indiv) if time_gains_indiv else np.nan
            report["Average Delay (minutes)"] = np.mean(delays_indiv) if delays_indiv else np.nan
            report["Std Delay (minutes)"] = np.std(delays_indiv) if delays_indiv else np.nan
            all_delays_for_sequences = delays_indiv # Store all for optional return
        else: # Should not happen if shape check passed
            report["Time Gain (minutes)"] = np.nan
            report["Delay (minutes)"] = np.nan

    # --- Scatter Plot ---
    if plot_scatter:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=10, color='blue', label='Predictions')
        min_val = min(y_true_flat.min(), y_pred_flat.min())
        max_val = max(y_true_flat.max(), y_pred_flat.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal (y=x)')
        plt.title('Actual vs. Predicted Glucose Values')
        plt.xlabel('Actual Glucose (mg/dL)')
        plt.ylabel('Predicted Glucose (mg/dL)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

    # --- Clarke Error Grid Plot and Zone Calculation ---
    if plot_ceg:
        ceg_zone_percentages = plot_clarke_error_grid(y_true_flat, y_pred_flat)
        report["CEG Zone A (%)"] = ceg_zone_percentages.get('A', 0.0)
        report["CEG Zone B (%)"] = ceg_zone_percentages.get('B', 0.0)
        report["CEG Zone C (%)"] = ceg_zone_percentages.get('C', 0.0)
        report["CEG Zone D (%)"] = ceg_zone_percentages.get('D', 0.0)
        report["CEG Zone E (%)"] = ceg_zone_percentages.get('E', 0.0)
        # You might want to also store the raw counts or the full percentages dict
        report["CEG_All_Zones_Percentages"] = ceg_zone_percentages


    if return_individual_delays:
        return report, all_delays_for_sequences
    else:
        return report
   

def plot_simple_delay_histograms(delay_data, model_names=None,
                                 title="Prediction Delays", xlabel="Delay (minutes)",
                                 n_bins=20, x_range=None, show_kde=False):
    """
    Plots simple histograms of prediction delays for comparison.

    Parameters:
    -----------
    delay_data : list of np.ndarray or np.ndarray
        - If a single np.ndarray: plots the distribution for one model.
        - If a list of np.ndarray: overlays histograms for multiple models.
          Each array contains per-sequence delays for a model.
    model_names : list of str, optional
        Names for each model. Required if `delay_data` is a list.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    n_bins : int, optional
        Number of bins for the histogram.
    x_range : tuple (min, max), optional
        X-axis range. If None, determined automatically.
    show_kde : bool, optional
        If True, overlays a Kernel Density Estimate plot for each histogram.
    """
    if not isinstance(delay_data, list):
        delay_data = [np.asarray(delay_data)]
    else:
        delay_data = [np.asarray(d) for d in delay_data]

    if model_names is None and len(delay_data) > 1:
        model_names = [f"Model {i+1}" for i in range(len(delay_data))]
    elif model_names is None and len(delay_data) == 1:
        model_names = ["Delays"]
    elif model_names and len(model_names) != len(delay_data):
        raise ValueError("model_names length must match delay_data length.")

    plt.figure(figsize=(10, 6))
    # sns.set_style("whitegrid") # Use a nice seaborn style
    colors = sns.color_palette("husl", n_colors=len(delay_data)) # Get distinct colors

    # Determine common x_range if not provided
    if x_range is None:
        all_delays_flat = np.concatenate([d[~np.isnan(d)] for d in delay_data if d is not None and len(d[~np.isnan(d)]) > 0])
        if len(all_delays_flat) > 0:
            min_val, max_val = np.min(all_delays_flat), np.max(all_delays_flat)
            padding = (max_val - min_val) * 0.05 if (max_val - min_val) > 0 else 1
            x_range = (min_val - padding, max_val + padding)
        else:
            print("No valid delay data to plot.")
            return


    for i, delays in enumerate(delay_data):
        delays_clean = delays[~np.isnan(delays)] # Remove NaNs
        if len(delays_clean) == 0:
            print(f"No valid data for {model_names[i]}. Skipping.")
            continue

        # Plot histogram
        plt.hist(delays_clean, bins=n_bins, range=x_range, density=True,
                 alpha=0.6, label=f"{model_names[i]}")

        # Optionally plot KDE
        if show_kde:
            sns.kdeplot(delays_clean, linewidth=1.5, fill=False)

        # Plot mean line
        mean_val = np.mean(delays_clean)
        plt.axvline(mean_val, linestyle='--', linewidth=1.5,
                    label=f"{model_names[i]} Mean: {mean_val:.2f} min")

    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Density", fontsize=12) # Density because density=True in hist
    if x_range:
        plt.xlim(x_range)
    plt.legend(fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show() 


def classification_report_extended(y_true, y_pred, y_prob=None, average=None, plot=True):
    """
    Generates a comprehensive classification report with metrics and optional plots.

    Parameters:
    - y_true (array-like): True target values.
    - y_pred (array-like): Predicted target values (class labels).
    - y_prob (array-like): Predicted probabilities for each class. Required for ROC-AUC and ROC curve.
    - average (str): Averaging method for multi-class data. Options: 'micro', 'macro', 'weighted', None.
    - plot (bool): Whether to plot the confusion matrix and ROC curve. Default is True.

    Returns:
    - report (dict): A dictionary containing classification metrics.
    """
    # Metrics for classification
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    if y_prob is not None:
        if len(np.unique(y_true)) == 2:  # Binary classification
            roc_auc = roc_auc_score(y_true, y_prob, average=average)
        else:  # Multi-class classification
            roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
    else:
        roc_auc = None

    # Detailed classification report
    detailed_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Constructing report
    report = {
        "Accuracy": accuracy,
        "Precision": list(precision),
        "Recall": list(recall),
        "F1-Score": list(f1),
    }
    if roc_auc is not None:
        report["ROC-AUC"] = roc_auc

    # Optional plots
    if plot:
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix')
        plt.show()

        # ROC Curve (Only for binary or multi-class probabilities provided)
        if y_prob is not None:
            if len(np.unique(y_true)) == 2:  # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend()
                plt.grid(True)
                plt.show()
            else:  # Multi-class
                print("ROC curve visualization is currently supported for binary classification only.")

    return report, pd.DataFrame(detailed_report).transpose()


def evaluate_model (model:StandardModel, X_train, y_train, X_val, y_val, X_test, y_test, plot = False, epochs = 30, early_stoping = 5, shuffle = False):
    model.fit(X_train,y_train,X_val,y_val,epochs,early_stoping,shuffle)
    y_clf_predict = model.predict(X_test)
    if model.classification:
        evaluation, df = classification_report_extended(y_test, y_clf_predict, plot=plot)
        return evaluation, df
    else:
        evaluation = regression_report(y_test, y_clf_predict, plot=plot)
        return evaluation

def f1score_hypo(results:dict) -> float:
    return float(results['F1-Score'][0])

def RMSE(results:dict) -> float:
    return float(results["Root Mean Squared Error (RMSE)"])

def R2SCORE(results:dict) -> float:
    return float(results["R^2 Score"])

def MAE(results:dict) -> float:
    return float(results["Mean Absolute Error (MAE)"])

# def tune_model_random (model_class:object, X_train, y_train, X_val, y_val, search_space:dict, criterion:function = f1score_hypo, max_try:int = 15, max_iter:int = 50):

#     max = 0.0 
#     iter_tries = 0
#     best = None

#     for iter in range(max_iter):
#         config = {keyword:random.choice(options) for keyword, options in search_space.items()}
#         model = model_class(**config)
#         evaluation = evaluate_classification_model(model, X_train, y_train, X_val, y_val)
#         score = criterion(evaluation)
#         if score > max:
#             best = config 
#             iter_tries = 0 
#             max = score 
#         else: 
#             iter_tries += 1 
#             if iter_tries == max_try:
#                 break
    
#     return best, max 

# ======================================== #
# Random Search for Hyper-parameter tuning #
# ======================================== #

def tune_model_random(
    model_class: object, 
    X_train, y_train, X_val, y_val, 
    search_space: dict, 
    epochs:int ,
    criterion:object = f1score_hypo, 
    max_try: int = 15, 
    max_iter: int = 50
):
    max_score = 0.0
    iter_tries = 0
    best_config = None
    best_evaluation = None

    for iteration in range(max_iter):
        config = {key: random.choice(values) for key, values in search_space.items()}
        model = model_class(**config)
        
        evaluation, _ = evaluate_model(model, X_train, y_train, X_val, y_val, epochs=epochs)
        score = criterion(evaluation)

        logger.info(f"Iteration {iteration+1}/{max_iter} - Config: {config} - Score: {score:.4f}")

        if score > max_score:
            best_config = config
            best_evaluation = evaluation
            iter_tries = 0
            max_score = score
            logger.info(f"New Best Model Found! Score: {max_score:.4f}")
        else:
            iter_tries += 1
        
        if iter_tries >= max_try:
            logger.info(f"Early stopping triggered after {max_try} unsuccessful attempts.")
            break

    logger.info(f"Best Model: {best_config} - Best Score: {max_score:.4f}")
    return best_config, best_evaluation

# ============================================ #
# Genetic Algorithm for Hyper-parameter tuning #
# ============================================ #

def random_init(max_population, init_range : dict):
    logger.info(f"Initializing random population of size {max_population}")
    population = []
    for i in range(max_population):
        individual = {}
        for keyword, range_value in init_range.items():
            if len(range_value) == 2 and isinstance(range_value[0], float) and isinstance(range_value[1], float):
                individual[keyword] = random.uniform(range_value[0], range_value[1])
            elif len(range_value) == 2 and isinstance(range_value[0], int) and isinstance(range_value[1], int):
                individual[keyword] = random.randint(range_value[0], range_value[1])
            else :
                if len(range_value) == 1:
                    individual[keyword] = range_value[0]
                else:
                    individual[keyword] = random.choice(range_value)
        population.append(individual)
        logger.info(f"Individual {i} created: {individual}")
    logger.info(f"Population initialization complete")
    return population

# defines a crossover function for parents to produce the new generation
def crossover(ind1:dict, scr1:float, ind2:dict, scr2:float):
    logger.info(f"Performing crossover between individuals with scores {scr1} and {scr2}")
    if scr1 > scr2:
        big_scr = scr1
        big_ind = ind1
        small_ind = ind2
    else:
        big_scr = scr2
        big_ind = ind2
        small_ind = ind1
    if scr1+scr2>0:
        proba = big_scr/(scr1+scr2)
    else:
        proba = 0.5
    logger.info(f"Probability for selecting genes from better individual: {proba}")

    new_ind = {}

    for big_chrom, small_chrom in zip(big_ind.items(), small_ind.values()):
        keyword, big_chrom = big_chrom
        if random.random() < proba:
            new_ind[keyword] = big_chrom
        else:
            new_ind[keyword] = small_chrom

    logger.info(f"Created new individual: {new_ind}")
    return new_ind

# defines a mutation function that actes on the new generation to add some variety and escape locale minimums
# in addition to discover new chromosomes. This is guided by a probability value
def mutation(ind:dict, proba=0.2, rates:dict=None, apply_mutation:bool=True):
    if not apply_mutation:
        return None
    logger.info(f"Attempting mutation with probability {proba}")
    if not isinstance(rates, list):
        rates = {keyword:0.1 for keyword in ind.keys()}
    assert len(ind) == len(rates), "The length of mutation rate doesn't correspond to the number of hyperparameters !!"

    for (keyword, value), rate in zip(ind.items(), rates.values()):
        logger.info(f"Considering mutation for {keyword}: {value} with rate {rate}")
        if random.random() < proba:
            old_value = value
            if isinstance(value, int):
                coeff = 1 if random.uniform(0,1) > 0.5 else -1
                ind[keyword] = int(value*(1 + coeff*rate))
            elif isinstance(value, float):
                coeff = 1 if random.uniform(0,1) > 0.5 else -1
                ind[keyword] = value*(1 + coeff*rate)
            elif isinstance(value, list):
                continue
            logger.info(f"Mutated {keyword}: {old_value} -> {ind[keyword]}")
    
    return ind

import multiprocessing as mp
def evaluate(ind, X_train, y_train, X_val, y_val, X_test, y_test, plot, epochs, early_stoping, shuffle, criterion, model_class):
        try:
            logger.info("Selecting parents from population")
            logger.info(f"Evaluating individual : {ind}")
            model = model_class(**ind)
            if model.classification:
                evaluation, _ = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, plot, epochs, early_stoping, shuffle)
            else:
                evaluation =  evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, plot, epochs, early_stoping, shuffle)
            score = criterion(evaluation)
            logger.info(f"Individual {ind} score: {score}")
        except Exception as e:
            print(f"Got error with {ind}:\n{e}")
            score = 0
        return score
    
def _generate_parameters(population, X_train, y_train, X_val, y_val, X_test, y_test, plot, 
                           epochs, early_stoping, shuffle, criterion, model_class):
    values = [[ind, X_train, y_train, X_val, y_val,  X_test, y_test, plot, 
                           epochs, early_stoping, shuffle, criterion, model_class] for ind in population]
    keywords_ = ["ind", "X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "plot", 
                           "epochs", "early_stoping", "shuffle", "criterion", "model_class"]
    parameters = []
    for value in values :  
        parameter = {keyword: param for keyword, param in zip(keywords_, value)}
        parameters.append(parameter)
    
    return values
# select parents from the population. Highest fitness measure (highest f1 score)
def select_parents(model_class, X_train, y_train, X_val, y_val, X_test, y_test, population:list, criterion:object = f1score_hypo,
                    parents=None, plot = False, epochs = 30, early_stoping = 5, shuffle = False, multiprocess:bool = True, ascending:bool=True):


    if multiprocess:
        parameters = _generate_parameters(population, X_train, y_train, X_val, y_val, X_test, y_test, plot, 
                           epochs, early_stoping, shuffle, criterion, model_class)
        with mp.Pool(len(population)) as pool:
            scr = pool.starmap(evaluate, parameters)
    else:
        scr = []
        for i, ind in enumerate(population):
            logger.info(f"Evaluating individual {i}: {ind}")
            scr.append (evaluate(ind, X_train, y_train, X_val, y_val, X_test, y_test, plot, epochs, early_stoping, shuffle, criterion, model_class))
    combined = list(zip(population, scr))
    if parents:
        logger.info(f"Adding previous parents to selection pool: {parents}")
        combined.extend(parents)
    sorted_combined = sorted(combined, key=lambda x:x[1], reverse=ascending)
    logger.info(f"Selected best parents with scores: {sorted_combined[0][1]}, {sorted_combined[1][1]}")
    return sorted_combined[:2] 

# the genetic search function that encapsulates all the previously defined functions.
def genetic_search(population_size, init_range, model_class, X_train,
                    y_train, X_val, y_val, X_test, y_test, criterion:object = f1score_hypo, ascending:bool=True,
                    parents=None, plot = False, epochs = 30, early_stoping = 5, shuffle = False,
                    stop_criterion=0.97, apply_mutation:bool=False, max_iterations=100, multiprocess:bool = True):
    logger.info("Starting genetic search")
    logger.info(f"Parameters: population_size={population_size}, stop_criterion={stop_criterion}, max_iterations={max_iterations}")
    
    logger.info("Initializing population...")
    population = random_init(population_size, init_range)
    logger.info("Population initialized!")
    
    parents = select_parents(model_class=model_class, X_train=X_train, 
                             y_train=y_train, X_val=X_val, y_val=y_val,
                             X_test=X_test, y_test=y_test, 
                             population=population, criterion=criterion,
                             parents=parents, plot=plot, epochs=epochs, 
                             early_stoping=early_stoping, shuffle=shuffle,
                             multiprocess=multiprocess, ascending=ascending)
    
    for generation in range(max_iterations):
        logger.info(f"Generation {generation}")
        logger.info(f"Best score: {parents[0][1]}")
        
        if parents[0][1] > stop_criterion and ascending:
            logger.info(f"Stop criterion met at generation {generation}")
            return parents[0]
        elif parents[0][1] < stop_criterion and not ascending:
            logger.info(f"Stop criterion met at generation {generation}")
            return parents[0]
        else:
            population = []
            logger.info(f"Creating new population of size {population_size-2}")
            for i in range(population_size-2):
                logger.info(f"Creating individual {i}")
                ind = crossover(parents[0][0], parents[0][1], parents[1][0], parents[1][1])
                mutated_ind = mutation(ind, apply_mutation=apply_mutation)
                if mutated_ind is None:
                    mutated_ind = random_init(1 , init_range)[0]
                population.append(mutated_ind)
                
        logger.info("Selecting parents for next generation")
        parents = select_parents(model_class=model_class, X_train=X_train, 
                             y_train=y_train, X_val=X_val, y_val=y_val,
                             X_test=X_test, y_test=y_test, 
                             population=population, criterion=criterion,
                             parents=parents, plot=plot, epochs=epochs, 
                             early_stoping=early_stoping, shuffle=shuffle,
                             multiprocess=multiprocess, ascending=ascending)
    
    logger.info("Max iterations reached!")
    logger.info(f"Best solution found: {parents[0][0]} with score: {parents[0][1]}")
    return parents


def reg_to_clf(y_reg, classes:list=[180, 70], index=[0, 1, 2, 3]):
    num_classes = len(classes) + 1
    y_clf = np.ones_like(y_reg) * int(num_classes)
    for num_class, limit in enumerate(classes):
        y_clf = np.where(y_reg < limit, int(num_classes - 1 - num_class), y_clf)

    if index is None: 
        y_clf = y_clf.min(axis = 1) 
    elif isinstance(index, list):
        y_clf = y_clf[:, index].min(axis = 1) 
    else:
        y_clf = y_clf[:, index]
    y_clf = y_clf - 1    
    return y_clf

def autoregression(model, X, iterations):
    for i in range (iterations):
        y_predict = model.predict(X[:, i:])
        y_predict = y_predict.reshape(-1,1)
        X = np.concatenate([X, y_predict], axis = 1)
    return X[:, -iterations:]


def explain_model(model:StandardModel, X_background, X_test):
    device = model.device
    X_background = torch.from_numpy(X_background).to(device=device)
    X_test = torch.from_numpy(X_test).to(device=device)
    # model = WrappedModel(model)
    explainer = shap.GradientExplainer(model, X_background)
    shap_values = explainer.shap_values(X_test)
    return shap_values 

def plot_shap_class(shap_values, class_target:int = 0, model_name:str = None):
    sns.set_theme(style="whitegrid")
    sns.set_palette("Set3")
    sns.set_context("notebook", font_scale=1.0)
    plt.figure(figsize=(10, 6))
    # Define feature groups
    feature_groups = {
        'CGM signal': 0,
        'Meal info': 1,
        'insuline dose': 2,
    }

    # Aggregate SHAP values for each group
    group_shap_values = {}
    for group_name, indices in feature_groups.items():
        group_shap_values[group_name] = shap_values[:, :, indices, class_target].sum(axis=1)
    
    # Convert to DataFrame for easier plotting
    group_shap_df = pd.DataFrame(group_shap_values)

    # Calculate mean absolute SHAP values for each group
    mean_abs_shap = group_shap_df.mean()

    # Plot
    mean_abs_shap.sort_values(ascending=True).plot.barh()
    if model_name is None:
        plt.title('Mean SHAP Values by Feature Group')
        plt.xlabel('Mean SHAP value')
        plt.ylabel('Feature Group')
        plt.tight_layout()
        plt.show()
    else:
        plt.title(f'{model_name}: Mean SHAP Values by Feature Group')
        plt.xlabel('Mean SHAP value')
        plt.ylabel('Feature Group')
        plt.tight_layout()
        plt.show()        
    

def plot_time_series_shap(shap_values, class_:int = 0, model_name:str = None):
    sns.set_theme(style="whitegrid")
    sns.set_palette("Set1")
    sns.set_context("notebook", font_scale=1.0)
    plt.figure(figsize=(10, 6))
    # Aggregate SHAP values for each timestamp
    group_shap_values = shap_values[:, :, :, class_].sum(axis=2)
    # Convert to DataFrame for easier plotting
    group_shap_df = pd.DataFrame(group_shap_values)

    # Calculate mean absolute SHAP values for each group
    mean_abs_shap = group_shap_df.mean().abs()

    # Plot
    mean_abs_shap.plot.barh()
    if model_name is None:
        plt.title('Mean Absolute SHAP Values by Time-step')
        plt.xlabel('Mean |SHAP value|')
        plt.ylabel('Time-step')
        plt.tight_layout()
        plt.show()
    else:
        plt.title(f'{model_name}: Mean Absolute SHAP Values by Time-step')
        plt.xlabel('Mean |SHAP value|')
        plt.ylabel('Time-step')
        plt.tight_layout()
        plt.show() 