from flask import Flask, jsonify, render_template_string, request
import os

import joblib
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, 'models', 'health_model.pkl')


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
  return bundle


store = normalize_model_store(load_model())
model = store['model']
feature_names = store['feature_names']
dataset_summary = store['dataset_summary']
metrics = store['metrics']

app = Flask(__name__)

HTML = '''
<!doctype html>
<html lang="pt">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Health Insurance Cost Predictor</title>
  <style>
    body { font-family: Inter, Arial, sans-serif; background: linear-gradient(135deg, #f8fafc, #eef2ff 55%, #e2e8f0); color: #0f172a; margin: 0; }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 40px 20px; }
    .hero { background: linear-gradient(135deg, #0f172a, #1e293b 70%); border: 1px solid rgba(15, 23, 42, 0.08); border-radius: 24px; padding: 32px; box-shadow: 0 20px 60px rgba(15,23,42,.12); color: #f8fafc; }
    .grid { display: grid; grid-template-columns: 1.1fr .9fr; gap: 24px; margin-top: 24px; }
    .card { background: #ffffff; border: 1px solid rgba(148, 163, 184, 0.28); border-radius: 22px; padding: 24px; box-shadow: 0 12px 30px rgba(15,23,42,.06); }
    h1 { margin: 0 0 8px; font-size: 2.4rem; }
    .sub { color: #cbd5e1; margin-bottom: 18px; }
    label { display:block; margin: 12px 0 6px; color: #334155; font-size: 0.95rem; }
    input, select { width: 100%; padding: 12px 14px; border-radius: 14px; border: 1px solid #cbd5e1; background: #ffffff; color: #0f172a; box-sizing: border-box; }
    button { margin-top: 18px; width: 100%; padding: 14px; border: 0; border-radius: 16px; background: linear-gradient(135deg, #2563eb, #0ea5e9); color: white; font-weight: 700; cursor: pointer; }
    .metric { background: linear-gradient(135deg, #1d4ed8, #06b6d4); border-radius: 20px; padding: 20px; margin-bottom: 14px; }
    .metric.red { background: linear-gradient(135deg, #dc2626, #f59e0b); }
    .big { font-size: 2.1rem; font-weight: 800; margin-top: 6px; }
    .hint { color: #475569; font-size: 0.92rem; }
    .table { width:100%; border-collapse: collapse; margin-top: 10px; }
    .table td { padding: 8px 0; border-bottom: 1px solid rgba(148, 163, 184, 0.2); }
    .pill { display:inline-block; padding: 6px 10px; border-radius:999px; background:#e2e8f0; color:#1e3a8a; margin-right: 8px; font-size: .85rem; }
    .summary { display:grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 18px; }
    .summary-card { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); border-radius: 18px; padding: 16px; }
    .summary-card .label { color: #cbd5e1; font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; }
    .summary-card .value { color: #fff; font-size: 1.5rem; font-weight: 800; margin-top: 4px; }
  </style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <span class="pill">Health</span><span class="pill">Regression</span><span class="pill">Streamlit + Flask</span>
    <h1>Health Insurance Cost Predictor</h1>
    <div class="sub">Previsão do custo anual de seguro de saúde com idade, BMI, filhos, tabagismo e região.</div>

    <div class="summary">
      <div class="summary-card"><div class="label">Rows</div><div class="value">{{ summary.rows|default('—') }}</div></div>
      <div class="summary-card"><div class="label">Target mean</div><div class="value">${{ '{:,.2f}'.format(summary.target_mean|default(0)) }}</div></div>
      <div class="summary-card"><div class="label">R²</div><div class="value">{{ '{:.4f}'.format(metrics.r2|default(0)) }}</div></div>
    </div>

    <div class="grid">
      <div class="card">
        <form method="post" action="/predict">
          <label>Age</label><input name="age" type="number" min="0" value="{{ form.age|default(29) }}">
          <label>Sex</label>
          <select name="sex">
            <option value="female" {% if form.sex == 'female' %}selected{% endif %}>female</option>
            <option value="male" {% if form.sex == 'male' %}selected{% endif %}>male</option>
          </select>
          <label>BMI</label><input name="bmi" type="number" step="0.01" min="0" value="{{ form.bmi|default(27.5) }}">
          <label>Children</label><input name="children" type="number" min="0" value="{{ form.children|default(0) }}">
          <label>Smoker</label>
          <select name="smoker">
            <option value="no" {% if form.smoker == 'no' %}selected{% endif %}>no</option>
            <option value="yes" {% if form.smoker == 'yes' %}selected{% endif %}>yes</option>
          </select>
          <label>Region</label>
          <select name="region">
            <option value="southwest" {% if form.region == 'southwest' %}selected{% endif %}>southwest</option>
            <option value="southeast" {% if form.region == 'southeast' %}selected{% endif %}>southeast</option>
            <option value="northwest" {% if form.region == 'northwest' %}selected{% endif %}>northwest</option>
            <option value="northeast" {% if form.region == 'northeast' %}selected{% endif %}>northeast</option>
          </select>
          <button type="submit">Predict Cost</button>
        </form>
      </div>

      <div class="card">
        {% if result is defined %}
          <div class="metric {% if result.segment == 'High' %}red{% endif %}">
            <div>Estimated annual cost</div>
            <div class="big">${{ '{:,.2f}'.format(result.prediction) }}</div>
          </div>
          <div class="hint">Segment: <strong>{{ result.segment }}</strong></div>
          <div class="hint">Reference band: {{ result.band_low }} - {{ result.band_high }} USD</div>
          <div class="hint" style="margin-top:10px;">This prediction uses the same pipeline as the training script: engineered features, imputation, one-hot encoding and Random Forest regression.</div>
          <table class="table">
            <tr><td>Age factor</td><td>{{ result.features.age }}</td></tr>
            <tr><td>BMI</td><td>{{ result.features.bmi }}</td></tr>
            <tr><td>Smoker</td><td>{{ result.features.smoker }}</td></tr>
            <tr><td>Region</td><td>{{ result.features.region }}</td></tr>
          </table>
        {% else %}
          <div class="metric">
            <div>Estimated annual cost</div>
            <div class="big">$0.00</div>
          </div>
          <div class="hint">Preenche os campos e carrega em Predict Cost.</div>
        {% endif %}
      </div>
    </div>
  </div>
</div>
</body>
</html>
'''


def build_input(form):
    return pd.DataFrame([
        {
            'age': float(form.get('age', 29)),
            'sex': form.get('sex', 'female'),
            'bmi': float(form.get('bmi', 27.5)),
            'children': float(form.get('children', 0)),
            'smoker': form.get('smoker', 'no'),
            'region': form.get('region', 'southwest'),
        }
    ])


def classify_cost(value: float) -> tuple[str, float, float]:
    if value < 10000:
        return 'Low', 0, 10000
    if value < 25000:
        return 'Medium', 10000, 25000
    return 'High', 25000, 100000


@app.route('/', methods=['GET'])
def index():
  return render_template_string(HTML, form={}, summary=dataset_summary, metrics=metrics)


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.form
    row = build_input(payload)
    predicted_cost = float(model.predict(row)[0])
    segment, band_low, band_high = classify_cost(predicted_cost)
    result = {
        'prediction': predicted_cost,
        'segment': segment,
        'band_low': f'{band_low:,.0f}',
        'band_high': f'{band_high:,.0f}',
        'features': row.iloc[0].to_dict(),
    }
    return render_template_string(HTML, result=result, form=payload, summary=dataset_summary, metrics=metrics)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json() or request.form
    row = build_input(payload)
    predicted_cost = float(model.predict(row)[0])
    segment, band_low, band_high = classify_cost(predicted_cost)
    return jsonify(
        {
            'predicted_cost': predicted_cost,
            'segment': segment,
            'band_low': band_low,
            'band_high': band_high,
        }
    )


if __name__ == '__main__':
    app.run(debug=False)
