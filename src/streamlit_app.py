import os

import joblib
import pandas as pd
import streamlit as st


ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, 'models', 'health_model.pkl')


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError('Modelo não encontrado. Execute health/src/train.py primeiro.')
    return joblib.load(MODEL_PATH)


def normalize_model_store(store):
    if isinstance(store, dict):
        bundle = dict(store)
    else:
        bundle = {'model': store}

    model = bundle.get('model')
    if model is None:
        model = bundle
        bundle['model'] = model

    if not bundle.get('feature_names') and hasattr(model, 'regressor_'):
        try:
            preprocessor = model.regressor_.named_steps['preprocessor']
            bundle['feature_names'] = list(preprocessor.get_feature_names_out())
        except Exception:
            bundle['feature_names'] = []

    bundle.setdefault('feature_names', [])
    bundle.setdefault('dataset_summary', {})
    bundle.setdefault('metrics', {})
    bundle.setdefault('feature_groups', {})
    bundle.setdefault('raw_features', [])
    bundle.setdefault('numeric_features', [])
    return bundle


store = normalize_model_store(load_model())
model = store['model']
feature_names = store['feature_names']
dataset_summary = store['dataset_summary']
metrics = store['metrics']
feature_groups = store['feature_groups']

st.set_page_config(page_title='Health Insurance Predictor', layout='wide', initial_sidebar_state='expanded')

st.markdown(
    '''
    <style>
    .main { background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%); }
    .hero {
        background: linear-gradient(135deg, rgba(15,23,42,.96), rgba(30,41,59,.96));
        border: 1px solid rgba(15,23,42,.10);
        padding: 28px;
        border-radius: 28px;
        box-shadow: 0 24px 60px rgba(15,23,42,.12);
    }
    .title { font-size: 2.8rem; font-weight: 800; color: #f8fafc; margin-bottom: .3rem; }
    .subtitle { color: #cbd5e1; font-size: 1.05rem; margin-bottom: 0; }
    .chip { display:inline-block; padding: 6px 12px; margin-right:8px; border-radius:999px; background:#1e293b; color:#bfdbfe; font-size:.85rem; }
    .section-card {
        border-radius: 20px;
        padding: 20px;
        background: #ffffff;
        border: 1px solid rgba(148,163,184,.25);
        box-shadow: 0 12px 32px rgba(15,23,42,.06);
    }
    .result-card {
        border-radius: 24px;
        padding: 24px;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px solid rgba(148,163,184,.18);
    }
    .cost-big { font-size: 2.6rem; font-weight: 800; color: #f8fafc; }
    .small-label { color: #64748b; font-size: .85rem; text-transform: uppercase; letter-spacing: .08em; }
    </style>
    ''',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><div class="chip">Health</div><div class="chip">Regression</div><div class="chip">Educational demo</div><div class="title">Health Insurance Cost Predictor</div><p class="subtitle">Modelo de regressão com features originais, engenharia de atributos e leitura interpretável do resultado.</p></div>',
    unsafe_allow_html=True,
)

top_a, top_b, top_c = st.columns(3)
with top_a:
    st.markdown(
        f'''
        <div class="section-card">
            <div class="small-label">Rows</div>
            <div style="font-size:1.9rem; font-weight:800; color:#0f172a;">{dataset_summary.get('rows', '—')}</div>
            <div style="color:#64748b;">Registos usados no treino</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
with top_b:
    st.markdown(
        f'''
        <div class="section-card">
            <div class="small-label">Target median</div>
            <div style="font-size:1.9rem; font-weight:800; color:#0f172a;">${dataset_summary.get('target_median', 0):,.2f}</div>
            <div style="color:#64748b;">Valor central das charges</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
with top_c:
    st.markdown(
        f'''
        <div class="section-card">
            <div class="small-label">R²</div>
            <div style="font-size:1.9rem; font-weight:800; color:#0f172a;">{metrics.get('r2', 0.0):.4f}</div>
            <div style="color:#64748b;">Desempenho no conjunto de teste</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

st.markdown('### Como o modelo foi construído')
st.markdown(
    '- As variáveis brutas vêm do CSV do tipo UCI: idade, sexo, BMI, filhos, tabagismo e região.\n'
    '- A função `engineer_features()` cria atributos derivados como obesidade, idade ao quadrado e interações.\n'
    '- Os dados numéricos recebem imputação por mediana e as categorias são codificadas com one-hot encoding.\n'
    '- O estimador final é um Random Forest regressivo com transformação log no target para estabilizar a escala.'
)

with st.expander('Detalhes do treino e do dataset', expanded=False):
    left, right = st.columns(2)
    with left:
        st.write('**Features originais**')
        st.write(feature_groups.get('raw', store['raw_features']))
        st.write('**Features numéricas usadas no pipeline**')
        st.write(feature_groups.get('engineered_numeric', store['numeric_features']))
    with right:
        st.write('**Estatísticas do target**')
        st.write({
            'min': dataset_summary.get('target_min'),
            'mean': dataset_summary.get('target_mean'),
            'median': dataset_summary.get('target_median'),
            'max': dataset_summary.get('target_max'),
        })
        st.write('**Métricas**')
        st.write({
            'MAE': metrics.get('mae'),
            'RMSE': metrics.get('rmse'),
            'R2': metrics.get('r2'),
        })

with st.sidebar:
    st.header('Input')
    age = st.slider('Age', 18, 64, 29)
    sex = st.selectbox('Sex', ['female', 'male'])
    bmi = st.slider('BMI', 15.0, 45.0, 27.5, 0.1)
    children = st.slider('Children', 0, 5, 0)
    smoker = st.selectbox('Smoker', ['no', 'yes'])
    region = st.selectbox('Region', ['southwest', 'southeast', 'northwest', 'northeast'])
    predict = st.button('Predict cost', width='stretch')


def build_row():
    return pd.DataFrame(
        [
            {
                'age': age,
                'sex': sex,
                'bmi': bmi,
                'children': children,
                'smoker': smoker,
                'region': region,
            }
        ]
    )


def band(value: float):
    if value < 10000:
        return 'Low', '#10b981'
    if value < 25000:
        return 'Medium', '#f59e0b'
    return 'High', '#ef4444'


if predict:
    row = build_row()
    prediction = float(model.predict(row)[0])
    segment, accent = band(prediction)
    st.markdown(
        f'''
        <div class="result-card" style="border-color:{accent};">
            <div style="color:#94a3b8;">Estimated annual cost</div>
            <div class="cost-big">${prediction:,.2f}</div>
            <div style="margin-top:8px; color:{accent}; font-weight:700;">Segment: {segment}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric('Age', age)
    with metric_col2:
        st.metric('BMI', f'{bmi:.1f}')
    with metric_col3:
        st.metric('Smoker', smoker)

    st.subheader('Feature importance')
    if hasattr(model, 'regressor_') and hasattr(model.regressor_.named_steps['regressor'], 'feature_importances_') and feature_names:
        regressor = model.regressor_.named_steps['regressor']
        importance = pd.Series(regressor.feature_importances_, index=feature_names).sort_values(ascending=False)
        st.bar_chart(importance.head(10))
    else:
        st.info('Feature importance is unavailable for this model file.')

    st.subheader('Input used')
    st.dataframe(row, width='stretch')
else:
    st.info('Use the controls in the sidebar and click Predict cost.')

st.markdown('---')
st.caption('Random Forest regression with feature engineering on age, BMI and smoker interactions.')
