import pandas as pd
import streamlit as st
import torch
from streamlit.delta_generator import DeltaGenerator

# -- PARAMETERS

dataset_vals = ['FGNet', 'RetinaMNIST']
model_vals = ['resnet18']
evaluation_target_vals = ['score_algorithm', 'loss_fn']
loss_fn_vals = ['CrossEntropy', 'TriangularCrossEntropy', 'OrdinalECOCDistance']
score_alg_vals = ['LAC']

# -- Streamlit Setup
st.set_page_config(page_title='Conformal Prediction', layout='wide', page_icon=':leg:')

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
param_evaluation_target = st.sidebar.segmented_control('Evaluation target', evaluation_target_vals, selection_mode='single',
                                                       default=evaluation_target_vals[0])

param_loss_fn: list[str] = []
param_score_alg: list[str] = []

if param_evaluation_target == 'score_algorithm':
    param_loss_fn = [st.sidebar.selectbox(
        'Loss function',
        loss_fn_vals,
    )]
    param_score_alg = st.sidebar.multiselect(
        'Score algorithm',
        score_alg_vals,
    )
elif param_evaluation_target == 'loss_fn':
    param_loss_fn = st.sidebar.multiselect(
        'Loss function',
        loss_fn_vals,
    )
    param_score_alg = [st.sidebar.selectbox(
        'Score algorithm',
        score_alg_vals,
    )]
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

@st.cache_resource
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
def load_dataset(dataset, hold_out_size):
    from datasets import loader
    return loader.load_dataset(dataset, hold_out_size)


if param_dataset == '' or param_model == '' or param_score_alg == [] or param_loss_fn == []:
    st.warning('Please input all parameters.')
    st.stop()

torch.classes.__path__ = []

cp_runner = load_cp_runner(param_dataset, param_model, param_score_alg, param_loss_fn, param_alpha)

if not cp_runner.has_run:
    if cp_runner.progress is None:
        st.write('The results for these parameters have not been evaluated yet.')
        if st.button('Start evaluation'):
            dataset = load_dataset(param_dataset, hold_out_size=param_hold_out_size)
            progress_bar = st.progress(0, text='Evaluation in progress...')
            cp_runner.run(dataset, progress_bar)
            st.rerun()
    else:
        st.write('Evaluating... Refresh the page to see the results.')
    st.stop()

# -- RESULTS

st.title('Results')

results = cp_runner.get_results()

df = pd.DataFrame(
    [(name, mean_width, coverage) for name, (_, _, mean_width, coverage) in results.items()],
    columns=['name', 'mean_width', 'coverage']
)
st.dataframe(df)
