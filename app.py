import os

import numpy as np
import pandas as pd
import streamlit as st
import torch

from metrics import get_metrics_across_reps
from util import frame_image_samples, st_narrow
from plots import (
    plot_overall_performance_comparison,
    plot_individual_metrics_grid,
    plot_size_stratified_coverage,
    plot_model_performance_metrics,
    plot_prediction_set_size_distribution
)

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
fig_overall = plot_overall_performance_comparison(df_performance, param_alpha)

with st_narrow():
    st.pyplot(fig_overall)

# Individual Metric Plots
st.subheader('Individual Metric Plots')
fig1, fig2, fig3, fig4 = plot_individual_metrics_grid(df, param_alpha)
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig1)
    st.pyplot(fig3)
with col2:
    st.pyplot(fig2)
    st.pyplot(fig4)

# 5. Size-Stratified Coverage (SSC)
st.subheader('Size-Stratified Coverage (SSC)')

fig5, ssc_df = plot_size_stratified_coverage(results, y_test, param_alpha)

if fig5 is not None:
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
fig_acc, fig_mae = plot_model_performance_metrics(df)
col_acc, col_mae = st.columns(2)
with col_acc:
    st.pyplot(fig_acc)
with col_mae:
    st.pyplot(fig_mae)

# 7. Prediction Set Size Distribution
st.subheader('Prediction Set Size Distribution')

fig7 = plot_prediction_set_size_distribution(results)

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
