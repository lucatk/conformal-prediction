import os

import numpy as np
import pandas as pd
import streamlit as st
import torch

from metrics import get_metrics_across_reps
from util import frame_image_samples, st_narrow
from matplotlib import pyplot as plt

# -- PARAMETERS

data_root = os.environ['DATA_ROOT'] if 'DATA_ROOT' in os.environ else '.'

dataset_vals = ['FGNet', 'Adience', 'RetinaMNIST']
model_vals = ['resnet18', 'resnet18-uni', 'resnet50']
loss_fn_vals = ['CrossEntropy', 'TriangularCrossEntropy', 'WeightedKappa', 'EMD']
score_alg_vals = ['LAC', 'APS', 'RAPS', 'RPS']

# -- Streamlit Setup
st.set_page_config(page_title='Conformal Prediction', layout='wide', page_icon=':parking:')

# Sidebar & File Initialisation for potential selection
st.sidebar.write('Configuration')

param_dataset = st.sidebar.selectbox(
    'Dataset',
    dataset_vals,
)
param_model = st.sidebar.selectbox(
    'Model',
    model_vals,
)

param_score_alg = st.sidebar.multiselect(
    'Score algorithm',
    score_alg_vals,
)
param_loss_fn = st.sidebar.multiselect(
    'Loss function',
    loss_fn_vals,
)
param_hold_out_size = st.sidebar.slider(
    'Hold out size',
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.01,
)
param_alpha = st.sidebar.slider(
    'Alpha',
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.01,
)
param_replication = st.sidebar.slider(
    'Replication',
    min_value=1,
    max_value=100,
    value=1,
    step=1,
)

torch.classes.__path__ = []


@st.cache_resource(hash_funcs={list[str]: lambda l: hash(frozenset(l))})
def load_cp_runner(dataset, model, score_alg, loss_fn, alpha):
    from cp_runner import CPRunner
    return CPRunner(
        dataset,
        model,
        score_alg,
        loss_fn,
        alpha,
        device='mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    )


@st.cache_resource
def load_dataset(dataset):
    from datasets import loader
    return loader.load_dataset(dataset, data_root)


if param_dataset == '' or param_model == '' or param_score_alg == [] or param_loss_fn == []:
    st.warning('Please input all parameters.')
    st.stop()

cp_runner = load_cp_runner(param_dataset, param_model, param_score_alg, param_loss_fn, param_alpha)

if not cp_runner.has_run:
    if cp_runner.progress is None:
        st.write('The results for these parameters have not been evaluated yet.')
        if st.button('Start evaluation'):
            dataset = load_dataset(param_dataset)
            progress_bar = st.progress(0, text='Evaluation in progress...')
            try:
                cp_runner.run(dataset, param_replication, progress_bar)
            except:
                st.error('An error occurred during evaluation. Please check the logs for more details.')
                if st.button('Retry'):
                    load_cp_runner.clear(param_dataset, param_model, param_score_alg, param_loss_fn, param_alpha)
                    st.rerun()
                raise
            st.rerun()
    elif cp_runner.has_error:
        st.error('An error occurred during evaluation. Please check the logs for more details.')
        if st.button('Retry'):
            load_cp_runner.clear(param_dataset, param_model, param_score_alg, param_loss_fn, param_alpha)
            st.rerun()
    else:
        st.write('Evaluating... Refresh the page to see the results.')
    st.stop()

# -- RESULTS
X_test, y_test = cp_runner.dataset.get_test_data()

st.subheader('Metrics')

results = cp_runner.get_results()

# Get metrics for all methods
metrics_data = []
for name, pred_results in results.items():
    metrics = get_metrics_across_reps(y_test, pred_results)
    loss_fn, score_alg = name.split('_', 1)
    metrics_data.append({
        'loss_fn': loss_fn,
        'score_alg': score_alg,
        **metrics._asdict()
    })

df = pd.DataFrame(metrics_data)

# Create performance score and sort by it
df['performance_score'] = (
    df['coverage'] -  # Higher is better
    df['mean_width'] -  # Lower is better, so subtract
    df['non_contiguous_percentage']  # Lower is better, so subtract
)
df = df.sort_values('performance_score', ascending=False)

# Round performance score for display
df['performance_score'] = df['performance_score'].round(3)

st.dataframe(
    df,
    column_config={
        'loss_fn': st.column_config.TextColumn('Loss Function', pinned=True),
        'score_alg': st.column_config.TextColumn('Score Algorithm', pinned=True),
        'coverage': st.column_config.NumberColumn('Coverage', format='%.3f', help='Classification Coverage Score (target: 1-α)'),
        'mean_width': st.column_config.NumberColumn('Mean Width', format='%.3f', help='Classification Mean Width Score'),
        'mean_range': st.column_config.NumberColumn('Mean Range', format='%.3f', help='Mean Interval Range (Regression Mean Width Score)'),
        'mean_gaps': st.column_config.NumberColumn('Mean Gaps', format='%.3f', help='Mean Gaps within Prediction Set'),
        'pred_set_mae': st.column_config.NumberColumn('Pred Set MAE', format='%.3f', help='Mean Absolute Error of Prediction Set'),
        'accuracy': st.column_config.NumberColumn('Accuracy', format='%.3f', help='Classification Accuracy'),
        'mae': st.column_config.NumberColumn('MAE', format='%.3f', help='Mean Absolute Error'),
        'non_contiguous_percentage': st.column_config.NumberColumn('Non-contiguous %', format='%.3f', help='Percentage of Non-Contiguous Prediction Sets'),
        'performance_score': st.column_config.NumberColumn('Performance Score', format='%.3f', help='Overall performance score (higher is better)'),
    }
)

# Overall Performance Comparison
st.subheader('Overall Performance Comparison')

# Create a performance score for each method
# Higher coverage is better, lower mean_width is better, lower non_contiguous_percentage is better
df_performance = df.copy()
df_performance['performance_score'] = (
    df_performance['coverage'] -  # Higher is better
    df_performance['mean_width'] -  # Lower is better, so subtract
    df_performance['non_contiguous_percentage']  # Lower is better, so subtract
)

# Sort by performance score (descending)
df_performance = df_performance.sort_values('performance_score', ascending=False)

# Create the comparison plot with dual y-axes
fig_overall, ax_overall = plt.subplots(figsize=(12, 8))
ax_overall_twin = ax_overall.twinx()  # Create second y-axis

# Create x-axis positions
x_pos_comp = np.arange(len(df_performance))

# Plot coverage and non-contiguous percentage on left y-axis (0-1 scale)
ax_overall.plot(x_pos_comp, df_performance['coverage'], 
         color='green', marker='o', label='Coverage', linewidth=2, markersize=8)
ax_overall.plot(x_pos_comp, df_performance['non_contiguous_percentage'], 
         color='red', marker='^', label='Non-contiguous %', linewidth=2, markersize=8)

# Plot mean width on right y-axis (different scale)
ax_overall_twin.plot(x_pos_comp, df_performance['mean_width'], 
         color='purple', marker='s', label='Mean Width', linewidth=2, markersize=8)

# Add target coverage line on left axis
ax_overall.axhline(y=1-param_alpha, color='green', linestyle='--', alpha=0.7, 
        label=f'Target Coverage (1-α = {1-param_alpha:.2f})')

# Customize the plot
ax_overall.set_xlabel('Methods (ordered by performance)', fontsize=14)
ax_overall.set_ylabel('Coverage / Non-contiguous %', color='black', fontsize=14)
ax_overall_twin.set_ylabel('Mean Width', color='purple', fontsize=14)

ax_overall.set_title('Overall Performance Comparison\n(Methods ordered by: max coverage, min mean width, min non-contiguous %)', fontsize=16)
ax_overall.set_xticks(x_pos_comp)
ax_overall.set_xticklabels([f"{row['loss_fn']}_{row['score_alg']}" for _, row in df_performance.iterrows()], 
                rotation=45, ha='right', fontsize=12)

# Set y-axis limits and colors
ax_overall.set_ylim(0, 1)  # Coverage and percentage are 0-1
ax_overall.tick_params(axis='y', labelcolor='black', labelsize=12)

# Set mean width axis color to purple
ax_overall_twin.tick_params(axis='y', labelcolor='purple', labelsize=12)

# Add legends
lines1, labels1 = ax_overall.get_legend_handles_labels()
lines2, labels2 = ax_overall_twin.get_legend_handles_labels()
ax_overall.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)

ax_overall.grid(axis='y', linestyle='--', alpha=0.7)

# Add performance scores as text annotations
for i, (_, row) in enumerate(df_performance.iterrows()):
    ax_overall.annotate(f'Score: {row["performance_score"]:.2f}', 
            xy=(i, row['coverage']), xytext=(5, 5),
            textcoords='offset points', fontsize=10, alpha=0.7)

plt.tight_layout()

with st_narrow():
    st.pyplot(fig_overall)

# Create comprehensive plots
st.subheader('Individual Metric Plots')

x_pos = np.arange(len(df))

# Create 2x2 grid using Streamlit columns
col1, col2 = st.columns(2)

with col1:
    # 1. Classification Mean Width Score (CMWS) - Mean Prediction Set Size
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x_pos, df['mean_width'], alpha=0.7, color='purple')
    ax1.set_xlabel('Methods', fontsize=12)
    ax1.set_ylabel('Mean Prediction Set Size', fontsize=12)
    ax1.set_title('Classification Mean Width Score (CMWS)', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{row['loss_fn']}_{row['score_alg']}" for _, row in df.iterrows()], rotation=45, ha='right', fontsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig1)
    
    # 3. Classification Coverage Score (CCS)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.bar(x_pos, df['coverage'], alpha=0.7, color='green')
    ax3.axhline(y=1-param_alpha, color='green', linestyle='--', label=f'Target Coverage (1-α = {1-param_alpha:.2f})')
    ax3.set_xlabel('Methods', fontsize=12)
    ax3.set_ylabel('Coverage', fontsize=12)
    ax3.set_title('Classification Coverage Score (CCS)', fontsize=14)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{row['loss_fn']}_{row['score_alg']}" for _, row in df.iterrows()], rotation=45, ha='right', fontsize=10)
    ax3.tick_params(axis='y', labelsize=10)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig3)

with col2:
    # 2. Regression Mean Width Score (RMWS) - Mean Interval Size/Range
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(x_pos, df['mean_range'], alpha=0.7, color='orange')
    ax2.set_xlabel('Methods', fontsize=12)
    ax2.set_ylabel('Mean Interval Size/Range', fontsize=12)
    ax2.set_title('Regression Mean Width Score (RMWS)', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{row['loss_fn']}_{row['score_alg']}" for _, row in df.iterrows()], rotation=45, ha='right', fontsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig2)
    
    # 4. Non-contiguous sets percentage
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.bar(x_pos, df['non_contiguous_percentage'], alpha=0.7, color='red')
    ax4.set_xlabel('Methods', fontsize=12)
    ax4.set_ylabel('Non-contiguous Sets (ratio)', fontsize=12)
    ax4.set_title('Non-contiguous Prediction Sets', fontsize=14)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"{row['loss_fn']}_{row['score_alg']}" for _, row in df.iterrows()], rotation=45, ha='right', fontsize=10)
    ax4.tick_params(axis='y', labelsize=10)
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig4)

# 5. Size-Stratified Coverage (SSC)
st.subheader('Size-Stratified Coverage (SSC)')

# Get all unique set sizes across all methods
all_sizes = set()
for name, pred_results in results.items():
    metrics = get_metrics_across_reps(y_test, pred_results)
    all_sizes.update(metrics.size_stratified_coverage.keys())

if all_sizes:
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
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(all_sizes))
    bar_width = 0.8 / len(ssc_df)
    
    for i, (_, row) in enumerate(ssc_df.iterrows()):
        coverages = [row[f'size_{size}'] for size in all_sizes]
        ax5.bar(x_pos + i * bar_width - (bar_width * len(ssc_df)) / 2, 
                coverages, bar_width, label=row['method'], alpha=0.7)
    
    ax5.set_xlabel('Prediction Set Size', fontsize=12)
    ax5.set_ylabel('Coverage (ratio)', fontsize=12)
    ax5.set_title('Size-Stratified Coverage (SSC)', fontsize=14)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(all_sizes, fontsize=10)
    ax5.tick_params(axis='y', labelsize=10)
    ax5.axhline(y=1-param_alpha, color='green', linestyle='--', 
                label=f'Target Coverage (1-α = {1-param_alpha:.2f})')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax5.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    with st_narrow():
        st.pyplot(fig5)
    
    # Display SSC table
    st.write("Size-Stratified Coverage Details:")
    ssc_display_df = ssc_df.copy()
    ssc_display_df = ssc_display_df.set_index('method')
    ssc_display_df.columns = [f'Size {col.split("_")[1]}' for col in ssc_display_df.columns]
    st.dataframe(ssc_display_df)

# 6. Model Performance Metrics
st.subheader('Model Performance Metrics')

x_pos = np.arange(len(df))  # Ensure this matches the number of methods

# Create 2-column layout using Streamlit
col_acc, col_mae = st.columns(2)

with col_acc:
    # Accuracy plot
    fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
    ax_acc.bar(x_pos, df['accuracy'], alpha=0.7, color='blue')
    ax_acc.set_xlabel('Methods', fontsize=12)
    ax_acc.set_ylabel('Accuracy', fontsize=12)
    ax_acc.set_title('Classification Accuracy', fontsize=14)
    ax_acc.set_xticks(x_pos)
    ax_acc.set_xticklabels([f"{row['loss_fn']}_{row['score_alg']}" for _, row in df.iterrows()], rotation=45, ha='right', fontsize=10)
    ax_acc.tick_params(axis='y', labelsize=10)
    ax_acc.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig_acc)

with col_mae:
    # MAE plot
    fig_mae, ax_mae = plt.subplots(figsize=(8, 5))
    ax_mae.bar(x_pos, df['mae'], alpha=0.7, color='red')
    ax_mae.set_xlabel('Methods', fontsize=12)
    ax_mae.set_ylabel('Mean Absolute Error', fontsize=12)
    ax_mae.set_title('Mean Absolute Error (MAE)', fontsize=14)
    ax_mae.set_xticks(x_pos)
    ax_mae.set_xticklabels([f"{row['loss_fn']}_{row['score_alg']}" for _, row in df.iterrows()], rotation=45, ha='right', fontsize=10)
    ax_mae.tick_params(axis='y', labelsize=10)
    ax_mae.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig_mae)

# 7. Prediction Set Size Distribution
st.subheader('Prediction Set Size Distribution')

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
fig7 = plt.figure(figsize=(10, 6))
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

with st_narrow():
    st.pyplot(fig7, use_container_width=False)

if param_replication == 1:
    st.subheader('Samples')

    samples_df = frame_image_samples(
        X_test,
        y_test,
        {name: pred[0][1] for name, pred in results.items()},
        cp_runner.dataset.get_class_labels()
    )

    st.dataframe(
        samples_df,
        column_config={
            'image': st.column_config.ImageColumn(),
        },
        row_height=64,
        height=600,
    )
