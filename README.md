# Health Insurance Cost Predictor

Projeto de machine learning para prever o custo anual de seguro de saúde com base em idade, sexo, BMI, número de filhos, tabagismo e região.

O projeto foi reorganizado para ficar mais próximo de uma apresentação académica: o treino guarda metadados do dataset e das métricas, e a interface mostra um resumo do problema, do pipeline e do resultado de forma mais explicativa.

## Problema

Prever o valor anual de `charges` a partir de variáveis estruturadas do paciente.

## Dados

- Fonte: `data/insurance.csv`
- Features brutas: `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- Target: `charges`

## Fluxo

1. Ler o CSV e validar as colunas obrigatórias.
2. Criar features derivadas em `src/feature_utils.py`.
3. Aplicar imputação e one-hot encoding no pipeline.
4. Treinar um Random Forest regressivo com transformação log no target.
5. Guardar o modelo e metadados em `models/health_model.pkl`.
6. Mostrar métricas, resumo do dataset e previsões na interface Streamlit.

## Estrutura

- `data/insurance.csv` - dataset base
- `src/train.py` - treino do modelo e exportação do artefacto
- `src/feature_utils.py` - engenharia de atributos
- `src/app.py` - demo Flask
- `src/streamlit_app.py` - interface Streamlit explicativa
- `models/` - modelo treinado

## Como correr

1. Treinar o modelo:

```bash
python src/train.py
```

2. Abrir a app Flask:

```bash
python src/app.py
```

3. Abrir a app Streamlit:

```bash
streamlit run src/streamlit_app.py
```

## Objetivo do projeto

O modelo prevê o custo anual estimado do seguro de saúde em dólares. A interface mostra também uma interpretação simples do resultado em faixas de custo, além de estatísticas do conjunto e importância das features para facilitar a demonstração.
