import os
import pickle

import pandas as pd
import streamlit as st
import torch
from streamlit.runtime.scriptrunner_utils.script_run_context import add_script_run_ctx

from datasets.adience import AdienceDataset
from datasets.fgnet import FGNetDataset
from datasets.retina_mnist import RetinaMNISTDataset
from metrics import get_metrics_across_reps
from util import frame_image_samples, st_narrow
from plots import (
    plot_overall_performance_comparison,
    plot_individual_metrics_grid,
    plot_size_stratified_coverage,
    plot_model_performance_metrics,
    plot_prediction_set_size_distribution,
)

from dotenv import load_dotenv

load_dotenv()

# -- PARAMETERS

runpod_token = os.getenv('RUNPOD_API_KEY')
runpod_pod_id = os.getenv('RUNPOD_POD_ID')

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
param_model = st.sidebar.multiselect(
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
alpha_presets = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
param_alpha = st.sidebar.multiselect(
    'Alpha',
    alpha_presets,
    default=[0.2],
)
param_replication = st.sidebar.slider(
    'Replication',
    min_value=1,
    max_value=100,
    value=1,
    step=1,
)

torch.classes.__path__ = []


def has_runpod_creds():
    return runpod_token is not None and runpod_pod_id is not None


@st.cache_resource(hash_funcs={
    FGNetDataset: lambda d: d.name,
    AdienceDataset: lambda d: d.name,
    RetinaMNISTDataset: lambda d: d.name,
    list[str]: lambda l: hash(frozenset(l))
})
def load_cp_runner(dataset, model, score_alg, loss_fn, alpha, replication):
    from cp_runner import CPRunner
    return CPRunner(
        dataset,
        model,
        score_alg,
        loss_fn,
        alpha,
        replication,
        device='mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    )


@st.cache_resource
def load_dataset(dataset):
    from datasets import loader
    return loader.load_dataset(dataset, data_root)


# Import/Export Section
st.sidebar.write('---')

dataset = None
cp_runner = None

# Import button
uploaded_file = st.sidebar.file_uploader(
    "Import Results",
    type=['pkl'],
    help='Upload a saved results file (.pkl)'
)

if uploaded_file is not None:
    try:
        save_data = pickle.loads(uploaded_file.getbuffer())

        dataset = load_dataset(save_data['dataset_name'])

        # Create a new CPRunner instance from the uploaded file
        from cp_runner import CPRunner

        cp_runner = CPRunner.from_save(save_data, dataset)

        st.sidebar.success(f"✅ Imported results from {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"❌ Error importing results: {str(e)}")
        cp_runner = None

# If no import, create CPRunner normally
if cp_runner is None:
    # Parameter validation
    if param_dataset == '' or param_model == [] or param_score_alg == [] or param_loss_fn == []:
        st.warning('Please input all parameters.')
        st.stop()
    dataset = load_dataset(param_dataset)
    cp_runner = load_cp_runner(dataset, param_model, param_score_alg, param_loss_fn, param_alpha, param_replication)

# Export button (enabled if results exist)
if cp_runner.has_run and not cp_runner.has_error:
    file_data, filename = cp_runner.export_results()
    st.sidebar.download_button(
        label="Export Results",
        data=file_data,
        file_name=filename,
        mime="application/octet-stream",
        help='Download current results as .pkl file'
    )

if not cp_runner.has_run:
    if cp_runner.progress is None:
        st.write('The results for these parameters have not been evaluated yet.')
        terminate_after_run = st.checkbox("Terminate after run and export results to disk") if has_runpod_creds() else False
        if st.button('Start evaluation'):
            cp_runner.set_terminate_after_run(terminate_after_run)

            add_script_run_ctx(cp_runner)
            cp_runner.start()
            cp_runner.join()
            st.rerun()
    elif cp_runner.has_error:
        st.error('An error occurred during evaluation. Please check the logs for more details.')
        if st.button('Retry'):
            load_cp_runner.clear(dataset, param_model, param_score_alg, param_loss_fn, param_alpha, param_replication)
            st.rerun()
    else:
        st.write('Evaluating... See logs for progress.')
        cp_runner.join()
        st.rerun()
    st.stop()

# -- RESULTS
X_test, y_test = cp_runner.dataset.get_test_data()
results = cp_runner.get_results()


# Helper function to calculate performance score
def calculate_performance_score(df_subset):
    """Calculate performance score for a subset of data."""
    if len(df_subset) == 0:
        return df_subset

    df_subset = df_subset.copy()
    df_subset['performance_score'] = (
                                             df_subset['coverage'] +  # Higher is better (already 0-1)
                                             (1 - df_subset['mean_width'] / df_subset[
                                                 'mean_width'].max()) +  # Normalize mean_width to 0-1, lower is better
                                             (1 - df_subset['non_contiguous_percentage'])
                                     # Non-contiguous % to 0-1, lower is better
                                     ) / 3  # Average of the three normalized metrics
    df_subset = df_subset.sort_values('performance_score', ascending=False)
    df_subset['performance_score'] = df_subset['performance_score'].round(5)
    return df_subset


# Helper function to display metrics dataframe
def display_metrics_dataframe(df_subset):
    """Display metrics dataframe with consistent column configuration."""
    st.dataframe(
        df_subset,
        column_config={
            'model': st.column_config.TextColumn('Model', pinned=True),
            'loss_fn': st.column_config.TextColumn('Loss Function', pinned=True),
            'score_alg': st.column_config.TextColumn('Score Algorithm', pinned=True),
            'alpha': None,
            'coverage': st.column_config.NumberColumn('Coverage', format='%.3f',
                                                      help='Classification Coverage Score (target: 1-α)'),
            'mean_width': st.column_config.NumberColumn('Mean Width', format='%.3f',
                                                        help='Classification Mean Width Score'),
            'mean_range': st.column_config.NumberColumn('Mean Range', format='%.3f',
                                                        help='Mean Interval Range (Regression Mean Width Score)'),
            'mean_gaps': st.column_config.NumberColumn('Mean Gaps', format='%.3f',
                                                       help='Mean Gaps within Prediction Set'),
            'pred_set_mae': st.column_config.NumberColumn('Pred Set MAE', format='%.3f',
                                                          help='Mean Absolute Error of Prediction Set'),
            'accuracy': st.column_config.NumberColumn('Accuracy', format='%.3f', help='Classification Accuracy'),
            'mae': st.column_config.NumberColumn('MAE', format='%.3f', help='Mean Absolute Error'),
            'qwk': st.column_config.NumberColumn('QWK', format='%.3f', help='Quadratic Weighted Kappa'),
            'non_contiguous_percentage': st.column_config.NumberColumn('Non-contiguous %', format='%.3f',
                                                                       help='Percentage of Non-Contiguous Prediction Sets'),
            'performance_score': st.column_config.NumberColumn('Performance Score', format='%.3f',
                                                               help='Overall performance score (higher is better)'),
        },
        hide_index=True,
    )


# Helper function to display all plots for a given alpha
def display_alpha_plots(df_alpha, alpha_val, results, y_test):
    """Display all plots for a specific alpha value."""
    # Overall Performance Comparison
    st.subheader('Overall Performance Comparison')
    df_performance_alpha = df_alpha.copy()
    fig_overall = plot_overall_performance_comparison(df_performance_alpha, alpha_val)
    with st_narrow():
        st.pyplot(fig_overall)

    # Individual Metric Plots
    st.subheader('Individual Metric Plots')
    fig1, fig2, fig3, fig4, fig5 = plot_individual_metrics_grid(df_alpha, alpha_val)
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
        st.pyplot(fig3)
        st.pyplot(fig5)
    with col2:
        st.pyplot(fig2)
        st.pyplot(fig4)

    # Size-Stratified Coverage
    st.subheader('Size-Stratified Coverage (SSC)')
    alpha_results = {name: pred for name, pred in results.items() if f'_alpha{alpha_val:.2f}' in name}
    fig5, ssc_df = plot_size_stratified_coverage(alpha_results, y_test, alpha_val)

    if fig5 is not None:
        with st_narrow():
            st.pyplot(fig5)

        # Display SSC table
        st.write("Size-Stratified Coverage Details:")
        ssc_display_df = ssc_df.copy()
        ssc_display_df = ssc_display_df.set_index('method')
        ssc_display_df.columns = [f'Size {col.split("_")[1]}' for col in ssc_display_df.columns]
        st.dataframe(ssc_display_df)

    # Model Performance Metrics
    st.subheader('Model Performance Metrics')
    fig_acc, fig_mae = plot_model_performance_metrics(df_alpha)
    col_acc, col_mae = st.columns(2)
    with col_acc:
        st.pyplot(fig_acc)
    with col_mae:
        st.pyplot(fig_mae)

    # Prediction Set Size Distribution
    st.subheader('Prediction Set Size Distribution')
    fig7 = plot_prediction_set_size_distribution(alpha_results)
    with st_narrow():
        st.pyplot(fig7, use_container_width=False)


# Get metrics for all methods
metrics_data = []
for name, pred_results in results.items():
    metrics = get_metrics_across_reps(y_test, pred_results)
    # Parse the format: model_loss_fn_score_alg_alphaX.XX
    alpha_part = name.split('_alpha')[-1]
    base_name = name.replace(f'_alpha{alpha_part}', '')
    parts = base_name.split('_', 2)
    model, loss_fn, score_alg = parts[0], parts[1], parts[2]
    alpha = float(alpha_part)

    metrics_data.append({
        'model': model,
        'loss_fn': loss_fn,
        'score_alg': score_alg,
        'alpha': alpha,
        **metrics._asdict()
    })

df = pd.DataFrame(metrics_data)
df = calculate_performance_score(df)

# Group by alpha and create tabs
unique_alphas = sorted(df['alpha'].unique())

if len(unique_alphas) > 1:
    # Create tabs for each alpha
    tab_names = [f"Alpha {alpha:.2f}" for alpha in unique_alphas]
    tabs = st.tabs(tab_names)

    for i, alpha_val in enumerate(unique_alphas):
        with tabs[i]:
            st.subheader(f'Metrics')

            # Filter data for this alpha and recalculate performance score
            df_alpha = df[df['alpha'] == alpha_val].copy()
            df_alpha = calculate_performance_score(df_alpha)

            display_metrics_dataframe(df_alpha)
            display_alpha_plots(df_alpha, alpha_val, results, y_test)
else:
    # Single alpha - display normally
    st.subheader('Metrics')
    display_metrics_dataframe(df)
    display_alpha_plots(df, df['alpha'].iloc[0], results, y_test)

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
