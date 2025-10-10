import os
import pickle

import matplotlib
import numpy
import streamlit as st
import torch
from streamlit.runtime.scriptrunner_utils.script_run_context import add_script_run_ctx

from cp_runner import CPRunner
from datasets.adience import AdienceDataset
from datasets.base_dataset import Dataset
from datasets.fgnet import FGNetDataset
from datasets.retina_mnist import RetinaMNISTDataset
from metrics import get_results_metrics
from util import frame_image_samples, st_narrow, render_plot_download_button
from plots import (
    plot_overall_performance_comparison,
    plot_individual_metrics_grid,
    plot_size_stratified_coverage,
    plot_model_performance_metrics,
    plot_prediction_set_size_distribution,
    plot_coverage_across_alphas,
    plot_classification_mean_width_across_alphas, plot_regression_mean_width_across_alphas,
    plot_non_contiguous_perc_across_alphas, plot_ssc_score_across_alphas,
)

from dotenv import load_dotenv

load_dotenv()

# -- PARAMETERS

data_root = os.environ['DATA_ROOT'] if 'DATA_ROOT' in os.environ else '.'

dataset_vals = ['FGNet', 'Adience', 'RetinaMNIST']
model_vals = ['resnet18', 'resnet18-uni', 'resnet50']
loss_fn_vals = ['CrossEntropy', 'TriangularCrossEntropy', 'WeightedKappa', 'EMD']
score_alg_vals = ['LAC', 'APS', 'RAPS', 'RPS']

# -- Streamlit Setup
st.set_page_config(page_title='Conformal Prediction', layout='wide', page_icon=':parking:')

# -- CSS
st.markdown(
    f"""
    <style>
        .stMultiSelect [data-baseweb=select] span{{
            max-width: 500px;
            # font-size: 0.8rem;
        }}
    </style>""",
    unsafe_allow_html=True,
)

# -- Torch setup
torch.classes.__path__ = []
seed = 1
numpy.random.seed(seed)
torch.manual_seed(seed)

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size' : 10,
    'text.usetex': False,
    'pgf.rcfonts': False
})


@st.cache_resource(hash_funcs={
    FGNetDataset: lambda d: d.name,
    AdienceDataset: lambda d: d.name,
    RetinaMNISTDataset: lambda d: d.name,
    list[str]: lambda l: hash(frozenset(l))
})
def load_cp_runner(dataset, model, score_alg, loss_fn, alpha, replication):
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
def load_dataset(dataset, hold_out_size):
    from datasets import loader
    return loader.load_dataset(dataset, hold_out_size, data_root)


@st.cache_resource
def load_cp_runner_from_save(uploaded_file):
    save_data = pickle.loads(uploaded_file.getbuffer())
    
    dataset = load_dataset(save_data['dataset_name'], 0.1)  # hold_out_size doesn't matter here
    
    from cp_runner import CPRunner
    return CPRunner.from_save(save_data, dataset)


@st.cache_data
def load_results_metrics(results, y_test, mode):
    return get_results_metrics(results, y_test, mode)


# Initialize dataset and cp_runner
dataset: Dataset = None
cp_runner: CPRunner = None

# Sidebar tabs
sidebar_tab1, sidebar_tab2, sidebar_tab3 = st.sidebar.tabs(['Evaluate', 'Import/Export', 'Results'])

# Evaluate tab content
with sidebar_tab1:
    param_dataset = st.selectbox(
        'Dataset',
        dataset_vals,
    )
    param_model = st.multiselect(
        'Model',
        model_vals,
    )

    param_score_alg = st.multiselect(
        'Score algorithm',
        score_alg_vals,
    )
    param_loss_fn = st.multiselect(
        'Loss function',
        loss_fn_vals,
    )
    param_hold_out_size = st.slider(
        'Hold out size',
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
    )
    alpha_presets = [0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3]
    param_alpha = st.multiselect(
        'Alpha',
        alpha_presets,
        default=[0.1],
    )
    param_replication = st.slider(
        'Replication',
        min_value=1,
        max_value=100,
        value=1,
        step=1,
    )

# Results tab content
with sidebar_tab2:
    # Import button
    uploaded_file = st.file_uploader(
        "Import Results",
        type=['pkl'],
        help='Upload a saved results file (.pkl)'
    )

    if uploaded_file is not None:
        try:
            # Create a new CPRunner instance from the uploaded file using cached function
            cp_runner = load_cp_runner_from_save(uploaded_file)

            st.success(f"✅ Imported results from {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Error importing results: {str(e)}")
            cp_runner = None

# If no import, create CPRunner normally
if cp_runner is None:
    # Parameter validation
    if param_dataset == '' or param_model == [] or param_score_alg == [] or param_loss_fn == []:
        st.warning('Please input all parameters.')
        st.stop()
    dataset = load_dataset(param_dataset, param_hold_out_size)
    cp_runner = load_cp_runner(dataset, param_model, param_score_alg, param_loss_fn, param_alpha, param_replication)

if not cp_runner.has_run:
    if cp_runner.progress is None:
        st.write('The results for these parameters have not been evaluated yet.')
        export_after_run = st.checkbox("Export results to disk after run finishes")
        if st.button('Start evaluation'):
            cp_runner.set_export_after_run(export_after_run)

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

    # Export button (enabled if results exist)
if cp_runner is not None and cp_runner.has_run and not cp_runner.has_error:
    with sidebar_tab2:
        if st.button('Prepare export'):
            file_data, filename = cp_runner.export_results()
            st.text(filename)
            st.download_button(
                label="Export Results",
                data=file_data,
                file_name=filename,
                mime="application/octet-stream",
                help='Download current results as .pkl file'
            )

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
            'name': None,
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


with sidebar_tab3:
    option_map = {
        0: "Per alpha",
        1: "Per method",
    }
    agg_mode = st.pills(
        "Aggregate",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        default=0,
        selection_mode="single",
    )
    if agg_mode == 0:
        df = load_results_metrics(results, y_test, 'mean')

        unique_alphas = sorted(df['alpha'].unique())
        selected_results = []
        alpha_val = df['alpha'].iloc[0]

        if len(unique_alphas) > 1:
            alpha_val = st.selectbox('Alpha', unique_alphas)
            df = df[df['alpha'] == alpha_val].copy()
        selected_results = st.multiselect(
            'Displayed results',
            df['name'].unique(),
            default=df['name'].unique(),
            format_func=lambda x: x.split('_alpha')[0],
        )

        df_filtered = df[df['name'].isin(selected_results)].copy()
        df_filtered = calculate_performance_score(df_filtered)
    elif agg_mode == 1:
        # pass
        df = load_results_metrics(results, y_test, 'collect')

        loss_fn = st.multiselect('Loss function', df['loss_fn'].unique(), key='metrics_loss_fn')
        score_alg = st.multiselect('Score algorithm', df['score_alg'].unique(), key='metrics_score_alg')

        df_filtered = df[(df['loss_fn'].isin(loss_fn)) & (df['score_alg'].isin(score_alg))].copy()
        df_filtered['method'] = df_filtered['loss_fn'] + '_' + df_filtered['score_alg']

if agg_mode == 0:
    st.subheader(f'Metrics')

    display_metrics_dataframe(df_filtered)
    display_alpha_plots(df_filtered, alpha_val, results, y_test)

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
elif agg_mode == 1:
    st.subheader('Plots')

    f_ccs = plot_coverage_across_alphas(df_filtered)
    st.pyplot(f_ccs)
    render_plot_download_button('ccs', loss_fn, score_alg, cp_runner.dataset.name, str(cp_runner.model), f_ccs)

    f_ssc_score = plot_ssc_score_across_alphas(df_filtered)
    st.pyplot(f_ssc_score)
    render_plot_download_button('ssc_score', loss_fn, score_alg, cp_runner.dataset.name, str(cp_runner.model), f_ssc_score)

    f_cmws = plot_classification_mean_width_across_alphas(df_filtered)
    st.pyplot(f_cmws)
    render_plot_download_button('cmws', loss_fn, score_alg, cp_runner.dataset.name, str(cp_runner.model), f_cmws)

    f_rmws = plot_regression_mean_width_across_alphas(df_filtered)
    st.pyplot(f_rmws)
    render_plot_download_button('rmws', loss_fn, score_alg, cp_runner.dataset.name, str(cp_runner.model), f_rmws)

    f_cv = plot_non_contiguous_perc_across_alphas(df_filtered)
    st.pyplot(f_cv)
    render_plot_download_button('cv', loss_fn, score_alg, cp_runner.dataset.name, str(cp_runner.model), f_cv)

