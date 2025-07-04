import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def plot_overall_performance_comparison(df_performance, param_alpha):
    """Create the overall performance comparison plot with three y-axes:
    - Left: Coverage (0-1, normal)
    - Right: Mean Width (range >1, normal)
    - Far right: Non-contiguous % (0-1, inverted, smaller is better at top)
    """
    import matplotlib as mpl
    fig, ax = plt.subplots(figsize=(12, 8))
    ax_mw = ax.twinx()  # Mean width axis (right)
    ax_nc = ax.twinx()  # Non-contiguous % axis (far right)

    # Offset the far right axis
    ax_nc.spines["right"].set_position(("axes", 1.12))
    ax_nc.spines["right"].set_visible(True)

    x_pos = np.arange(len(df_performance))

    # Plot coverage (left)
    p1, = ax.plot(x_pos, df_performance['coverage'], color='green', marker='o', label='Coverage', linewidth=2, markersize=8)
    ax.axhline(y=1-param_alpha, color='green', linestyle='--', alpha=0.7, label=f'Target Coverage (1-α = {1-param_alpha:.2f})')

    # Plot mean width (right)
    p2, = ax_mw.plot(x_pos, df_performance['mean_width'], color='purple', marker='s', label='Mean Width', linewidth=2, markersize=8)

    # Plot non-contiguous % (far right, inverted)
    p3, = ax_nc.plot(x_pos, df_performance['non_contiguous_percentage'], color='red', marker='^', label='Non-contiguous %', linewidth=2, markersize=8)
    ax_nc.set_ylim(1, 0)  # Invert so smaller is better at top

    # Axis labels
    ax.set_xlabel('Methods (ordered by performance)', fontsize=14)
    ax.set_ylabel('Coverage', color='green', fontsize=14)
    ax_mw.set_ylabel('Mean Width', color='purple', fontsize=14)
    ax_nc.set_ylabel('Non-contiguous %', color='red', fontsize=14)

    # Axis colors
    ax.tick_params(axis='y', labelcolor='green', labelsize=12)
    ax_mw.tick_params(axis='y', labelcolor='purple', labelsize=12)
    ax_nc.tick_params(axis='y', labelcolor='red', labelsize=12)

    # X axis
    ax.set_xticks(x_pos)
    # Compose two-line labels: method name and score
    xtick_labels = [f"{row['loss_fn']}_{row['score_alg']}\nScore: {row['performance_score']:.3f}" for _, row in df_performance.iterrows()]
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=12)
    ax.set_ylim(0, 1)

    # Legends
    lines = [p1, p2, p3]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', fontsize=12)

    ax.set_title('Overall Performance Comparison\n(Methods ordered by: max coverage, min mean width, min non-contiguous %)', fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig


def plot_classification_mean_width_score(df):
    """Create the Classification Mean Width Score (CMWS) plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(df))
    
    ax.bar(x_pos, df['mean_width'], alpha=0.7, color='purple')
    ax.set_xlabel('Methods', fontsize=12)
    ax.set_ylabel('Mean Prediction Set Size', fontsize=12)
    ax.set_title('Classification Mean Width Score (CMWS)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{row['loss_fn']}_{row['score_alg']}" for _, row in df.iterrows()], rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig


def plot_regression_mean_width_score(df):
    """Create the Regression Mean Width Score (RMWS) plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(df))
    
    ax.bar(x_pos, df['mean_range'], alpha=0.7, color='orange')
    ax.set_xlabel('Methods', fontsize=12)
    ax.set_ylabel('Mean Interval Size/Range', fontsize=12)
    ax.set_title('Regression Mean Width Score (RMWS)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{row['loss_fn']}_{row['score_alg']}" for _, row in df.iterrows()], rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig


def plot_classification_coverage_score(df, param_alpha):
    """Create the Classification Coverage Score (CCS) plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(df))
    
    ax.bar(x_pos, df['coverage'], alpha=0.7, color='green')
    ax.axhline(y=1-param_alpha, color='green', linestyle='--', label=f'Target Coverage (1-α = {1-param_alpha:.2f})')
    ax.set_xlabel('Methods', fontsize=12)
    ax.set_ylabel('Coverage', fontsize=12)
    ax.set_title('Classification Coverage Score (CCS)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{row['loss_fn']}_{row['score_alg']}" for _, row in df.iterrows()], rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig


def plot_non_contiguous_prediction_sets(df):
    """Create the Non-contiguous Prediction Sets plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(df))
    
    ax.bar(x_pos, df['non_contiguous_percentage'], alpha=0.7, color='red')
    ax.set_xlabel('Methods', fontsize=12)
    ax.set_ylabel('Non-contiguous Sets (ratio)', fontsize=12)
    ax.set_title('Non-contiguous Prediction Sets', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{row['loss_fn']}_{row['score_alg']}" for _, row in df.iterrows()], rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig


def plot_individual_metrics_grid(df, param_alpha):
    """Return the 4 individual metric plots as a tuple of figures for a 2x2 grid layout."""
    fig1 = plot_classification_mean_width_score(df)
    fig2 = plot_regression_mean_width_score(df)
    fig3 = plot_classification_coverage_score(df, param_alpha)
    fig4 = plot_non_contiguous_prediction_sets(df)
    return fig1, fig2, fig3, fig4


def plot_size_stratified_coverage(results, y_test, param_alpha):
    """Create the size-stratified coverage plot."""
    from metrics import get_metrics_across_reps
    
    # Get all unique set sizes across all methods
    all_sizes = set()
    for name, pred_results in results.items():
        metrics = get_metrics_across_reps(y_test, pred_results)
        all_sizes.update(metrics.size_stratified_coverage.keys())

    if not all_sizes:
        return None, None
    
    all_sizes = sorted(all_sizes)
    
    # Create data for plotting
    ssc_data = []
    for name, pred_results in results.items():
        metrics = get_metrics_across_reps(y_test, pred_results)
        row_data = {'method': name}
        for size in all_sizes:
            row_data[f'size_{size}'] = metrics.size_stratified_coverage.get(size, 0.0)
        ssc_data.append(row_data)
    
    ssc_df = pd.DataFrame(ssc_data)
    
    # Plot size-stratified coverage
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(all_sizes))
    bar_width = 0.8 / len(ssc_df)
    
    for i, (_, row) in enumerate(ssc_df.iterrows()):
        coverages = [row[f'size_{size}'] for size in all_sizes]
        ax.bar(x_pos + i * bar_width - (bar_width * len(ssc_df)) / 2, 
                coverages, bar_width, label=row['method'], alpha=0.7)
    
    ax.set_xlabel('Prediction Set Size', fontsize=12)
    ax.set_ylabel('Coverage (ratio)', fontsize=12)
    ax.set_title('Size-Stratified Coverage (SSC)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_sizes, fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.axhline(y=1-param_alpha, color='green', linestyle='--', 
                label=f'Target Coverage (1-α = {1-param_alpha:.2f})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    return fig, ssc_df


def plot_classification_accuracy(df):
    """Create the Classification Accuracy plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(df))
    
    ax.bar(x_pos, df['accuracy'], alpha=0.7, color='blue')
    ax.set_xlabel('Methods', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Classification Accuracy', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{row['loss_fn']}_{row['score_alg']}" for _, row in df.iterrows()], rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig


def plot_mean_absolute_error(df):
    """Create the Mean Absolute Error (MAE) plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(df))
    
    ax.bar(x_pos, df['mae'], alpha=0.7, color='red')
    ax.set_xlabel('Methods', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Mean Absolute Error (MAE)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{row['loss_fn']}_{row['score_alg']}" for _, row in df.iterrows()], rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig


def plot_model_performance_metrics(df):
    """Return the model performance metrics plots (Accuracy and MAE) as a tuple of figures."""
    fig_acc = plot_classification_accuracy(df)
    fig_mae = plot_mean_absolute_error(df)
    return fig_acc, fig_mae


def plot_prediction_set_size_distribution(results):
    """Create the prediction set size distribution plot."""
    predictor_names = list(results.keys())
    n_predictors = len(predictor_names)

    # Assume all methods have same number of classes
    n_classes = next(iter(results.values()))[0][1].shape[1]
    x = np.arange(1, n_classes + 1)  # possible set sizes

    hist_data = []
    for pred in predictor_names:
        pred_hists = []
        for _, y_pred_set in results[pred]:  # list of tuples
            pred_set = y_pred_set[:, :, 0]  # Drop alpha dim
            set_sizes = np.sum(pred_set, axis=1)
            hist, _ = np.histogram(set_sizes, bins=np.arange(0.5, n_classes + 1.5))
            pred_hists.append(hist)
        avg_hist = np.mean(np.stack(pred_hists), axis=0)
        hist_data.append(avg_hist)

    # Plot grouped histogram
    bar_width = 0.8 / n_predictors
    fig = plt.figure(figsize=(10, 6))
    for i, hist in enumerate(hist_data):
        plt.bar(x + i * bar_width - (bar_width * n_predictors) / 2, hist,
                width=bar_width, label=predictor_names[i])

    plt.xlabel('Prediction Set Size', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Set Size Distribution by Method', fontsize=14)
    plt.xticks(x, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    return fig 