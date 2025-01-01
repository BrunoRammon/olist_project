"""
This is a boilerplate pipeline 'monitoring'
generated using Kedro 0.19.10
"""

import os
from typing import Any, Callable, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
import shap
from olist_project.utils.utils import CustomLGBMClassifier

SOURCE_COLOR_MAP = {
    'ord': 'blue',
    'itm': 'green',
    'geo': 'orange',
    'sel': 'purple',
    'pay': 'yellow',
    'rev': 'red',
    'ctm': 'gray',
}

def create_partitions(df: pd.DataFrame) -> Dict[str, Any]:

    today = pd.Timestamp.now().strftime('%Y%m%d')
    if 'cohort' in df.columns:
        partitions = dict()
        for safra in sorted(df.cohort.unique()):
            partitions[str(int(safra))+'_'+today] = (
                df.query(f'cohort=={safra}').reset_index(drop=True)
            )
        return partitions
    else:
        return {
            today: df
        }

def concat_partitions(partitioned_input: Dict[str, Callable[[], Any]]) -> pd.DataFrame:
    """Concatenate input partitions into one pandas DataFrame.

    Args:
        partitioned_input: A dictionary with partition ids as keys and load functions as values.

    Returns:
        Pandas DataFrame representing a concatenation of all loaded partitions.
    """
    result = pd.DataFrame()

    for partition_key, partition_load_func in sorted(partitioned_input.items()):
        date_string = partition_key.split("_")[-1]
        partition_data = partition_load_func()  # load the actual partition data
        partition_data = (
            partition_data
            .assign(
                timestamp_partition = pd.to_datetime(date_string, format="%Y%m%d")
            )
        )
        # concat with existing result
        result = pd.concat([result, partition_data], ignore_index=True)

    result = (
        result
        .sort_values('timestamp_partition', ascending=True)
        .drop_duplicates(['seller_id', 'cohort'], keep='first')
        .drop(columns=['timestamp_partition'])
        .reset_index(drop=True)
    )

    return result

def _calc_roc_auc(df: pd.DataFrame,
                  target_name: str,
                  proba_name: str = 'proba'):
    if df[target_name].nunique() > 1:
        return roc_auc_score(df[target_name], df[proba_name])
    else:
        return 0

def _calc_ks(df: pd.DataFrame,
             target_name: str,
             proba_name: str = 'proba'):
    proba = df[proba_name]

    if df[target_name].nunique() > 1:
        return ks_2samp(proba[df[target_name] == 0],
                        proba[df[target_name] == 1]).statistic
    else:
        return 0

def model_evaluation(
    results: pd.DataFrame,
    spine: pd.DataFrame,
    target_name: str
)-> pd.DataFrame:
    """
    Calcula as performances auc e ks do modelo ao longo das safras
    """

    df_metrics_by_time = (
        spine
        .merge(results.filter(["seller_id", "cohort", "proba"]),
               on=["seller_id", "cohort"], how='inner')
        .groupby(["cohort"])
        .apply(lambda x:  pd.Series({'auc': _calc_roc_auc(x, target_name)*100,
                                     'ks': _calc_ks(x, target_name)*100}))
        .reset_index()
    )
    return df_metrics_by_time

#########################################################################################
################################# INPUT MONITORING ######################################
#########################################################################################

def _classify_period(safra, 
                     start_cohort: int = 202101,
                     split_cohort: int = 202401,
                     final_cohort: int = 202405):
    # Lógica para classificar o período (treino, teste, produção)
    conditions = [
        safra < split_cohort,  # Período de treino
        (split_cohort <= safra) & (safra <= final_cohort),  # Período de teste
        safra > final_cohort  # Período de produção
    ]
    choices = ['Treino', 'Teste', 'Produção']

    # Retorna a classificação baseada nas condições
    return np.select(conditions, choices, default='Produção')

def _calculate_metrics_by_period(
    df: pd.DataFrame,
    proba_column: str,
    target_column: str,
    period_column: str
) -> pd.DataFrame:

    df_metrics = (
        df
        .groupby(period_column, observed=True)
        .apply(lambda x: pd.Series({
            'auc': _calc_roc_auc(x, target_column, proba_column),
            'ks': _calc_ks(x, target_column, proba_column)
        }))
        .reset_index()
    )

    return df_metrics

def _generate_metrics_plots(
    metrics_by_month: pd.DataFrame,
    metrics_by_period: pd.DataFrame) -> go.Figure:

    metrics_by_month['cohort'] = pd.to_datetime(metrics_by_month['cohort'], format='%Y%m')

    # Criar subplots: 2 linhas e 1 coluna
    fig = sp.make_subplots(
        rows=2, cols=1,
        subplot_titles=("Performance por Mês", "Performance por Período"),
        vertical_spacing=0.1
    )

    # Gráfico de linha para auc e ks por Mês
    fig.add_trace(
        go.Scatter(x=metrics_by_month['cohort'], y=metrics_by_month['auc'] * 100,
                   mode='lines+markers', name='auc', line=dict(color='blue'), legendgroup = '1'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=metrics_by_month['cohort'], y=metrics_by_month['ks'] * 100,
                   mode='lines+markers', name='ks', line=dict(color='red'), legendgroup='1'),
        row=1, col=1
    )

    # Gráfico de barras para auc e ks por Período
    fig.add_trace(
        go.Bar(x=metrics_by_period['ks'] * 100, y=metrics_by_period['periodo'],
                name='ks', orientation='h', marker=dict(color='red'), legendgroup='2'),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(x=metrics_by_period['auc'] * 100, y=metrics_by_period['periodo'],
                name='auc', orientation='h', marker=dict(color='blue'), legendgroup='2'),
        row=2, col=1
    )

    fig.update_yaxes(title_text="Performance (%)", row=1, col=1)
    fig.update_xaxes(title_text="Performance (%)", row=2, col=1)

    # Atualizar o layout
    fig.update_layout(
        title_text='Monitoramento de Métricas de Performance',
        height=600,
        showlegend=True,
        legend_tracegroupgap = 200,
    )

    return fig

def _calculate_rating_metrics(
    df: pd.DataFrame,
    rating_column: str,
    target_column: str,
    period_column: str
) -> pd.DataFrame:

    # Calcular o total de registros por período
    total_volume = df.groupby(period_column, observed=True).size().reset_index(name='total_volume')

    # Calcular o volume e a inadimplência
    metrics_df = (
        df.groupby([period_column, rating_column], observed=True)
        .agg(
            volume=(target_column, 'size'),  # Contar o número de registros para cada rating por período
            inadimplencia=(target_column, 'mean')  # Calcular a média do target para cada rating por período
        )
        .reset_index()
    )

    # Juntar com o total para calcular o volume percentual
    metrics_df = metrics_df.merge(total_volume, on=period_column)
    metrics_df['volume'] = metrics_df['volume'] / metrics_df['total_volume']

    # Selecionar as colunas desejadas
    result_df = metrics_df[[period_column, rating_column, 'volume', 'inadimplencia']]

    return result_df

def _generate_rating_metrics_plots(
    rating_metrics_by_month: pd.DataFrame, rating_metrics_by_period: pd.DataFrame) -> go.Figure:    
    rating_metrics_by_month['cohort'] = pd.to_datetime(rating_metrics_by_month['cohort'],
                                                      format='%Y%m')
    unique_periods = rating_metrics_by_period['periodo'].unique()
    len_periods = len(unique_periods)
    # Criar subplots: 2 linhas e 1 coluna (parte de cima e parte de baixo)
    fig = sp.make_subplots(
        rows=2, cols=len_periods,
        specs=[[{"colspan": len_periods}]+[None]*(len_periods-1),
               [{"secondary_y": True}]*len_periods],
        subplot_titles=["Inadimplência por Rating - Mensal",]+list(unique_periods),
        vertical_spacing=0.1
    )

    # Parte de cima: Gráfico de linha para inadimplência por rating por mês
    for rating in sorted(rating_metrics_by_month['rating'].unique()):
        rating_data = rating_metrics_by_month[rating_metrics_by_month['rating'] == rating]
        fig.add_trace(
            go.Scatter(x=rating_data['cohort'], y=rating_data['inadimplencia'] * 100,
                       mode='lines+markers', name=f'{rating}', legendgroup='1'),
            row=1, col=1
        )
    fig.update_xaxes(title_text="Mês", row=1, col=1)
    fig.update_yaxes(title_text="Inadimplência (%)", row=1, col=1)

    # Parte de baixo: Gráficos para cada período

    for i, period in enumerate(unique_periods):
        period_data = (
            rating_metrics_by_period[rating_metrics_by_period['periodo'] == period]
            .sort_values('rating', ascending=False)
            .assign(
                volume_acumulado = lambda df: df.volume.cumsum()
            )
        )

        # Adicionar gráfico de barras para inadimplência
        fig.add_trace(
            go.Bar(x=period_data['rating'], y=period_data['inadimplencia'] * 100,
                   name='Inadimplência',
                   marker=dict(color='blue'),
                   showlegend=(i == 2), legendgroup='2'),
            row=2, col=int(i+1), secondary_y=False
        )

        # Adicionar gráfico de linha para volume
        fig.add_trace(
            go.Scatter(x=period_data['rating'], y=period_data['volume_acumulado'] * 100,
                       name='Volume Acumulado',
                       mode='lines+markers', 
                       line=dict(color='red'),
                       showlegend=(i == 2), legendgroup='2'),
            row=2, col=int(i+1), secondary_y=True
        )

        fig.update_xaxes(title_text="Rating", row=2, col=int(i+1))
        fig.update_yaxes(title_text="Inadimplência (%)", row=2, col=int(i+1), secondary_y=False)
        fig.update_yaxes(title_text="Volume Acumulado (%)", row=2, col=int(i+1),
                         secondary_y=True, range=[0,102])

    # Atualizar layout
    fig.update_layout(
        title_text='Monitoramento de Inadimplência e Volume por Rating',
        height=800,
        width=1400,
        showlegend=True,
        legend_tracegroupgap = 250,
    )

    return fig

def _define_numeric_bins_limits(
    df: pd.DataFrame, feature: str, num_bins: int = 5
):

    unique_values = df[feature].nunique()
    if unique_values < num_bins:
        num_bins = unique_values
        bins = pd.cut(df[feature].dropna(), bins=num_bins, retbins=True, duplicates='drop')[1]
    else:
        bins = pd.qcut(df[feature].dropna(), q=num_bins, retbins=True, duplicates='drop')[1]

    return bins

def _calculate_bins_per_feature(
    df: pd.DataFrame,
    feature: str,
    period_data: pd.DataFrame,
    num_bins: int = 5,
) -> pd.Series:

    if pd.api.types.is_numeric_dtype(df[feature]):
        bins_feature = _define_numeric_bins_limits(period_data, feature, num_bins)
        df['bins'] = pd.cut(df[feature], bins=bins_feature, labels=None, include_lowest=True)
        if df[feature].isna().sum() > 0:
            df['bins'] = df['bins'].astype('category')
            df['bins'] = df['bins'].cat.add_categories('Nulo')
            df['bins'] = df['bins'].fillna('Nulo')

    elif pd.api.types.is_categorical_dtype(df[feature]) or pd.api.types.is_object_dtype(df[feature]):
        df['bins'] = df[feature].astype(str)

    return df.bins


def _calculate_metrics_per_feature(
    df: pd.DataFrame,
    features: List[str],
    target_column: str,
    period_column: str,
    num_bins: int = 5,
    reference_period: str = 'Teste'
) -> pd.DataFrame:

    dfs_features = []

    period_data = df[df['periodo'] == reference_period]
    for feature in features:
        df['grupo'] = _calculate_bins_per_feature(df, feature, period_data, num_bins)

        total_volume = (
            df
            .groupby(period_column, observed=True)
            .agg(
                total_volume=(target_column, 'size'),
                total_volume_inad=(target_column, 'sum'),
            )
            .reset_index()
        )

        result_df = (
            df.groupby([period_column, 'grupo'], observed=True)
            .agg(
                volume=(target_column, 'size'),
                volume_inad=(target_column, 'sum'),
                inadimplencia=(target_column, 'mean')
            )
            .reset_index()
            .merge(total_volume, on=period_column)
            .assign(
                volume_not_inad = lambda df: (
                    (df.volume - df.volume_inad) / (df.total_volume - df.total_volume_inad)
                ),
                volume_inad = lambda df: df.volume_inad / df.total_volume_inad,
                volume = lambda df: df.volume / df.total_volume,
                WOE = lambda df: np.log(df.volume_not_inad/(df.volume_inad + 1e-6)),
                iv = lambda df: (
                    df.WOE * (df.volume_not_inad - df.volume_inad)
                ).replace({np.inf:np.nan,-np.inf:np.nan}).fillna(0),
                variavel = feature,
                grupo = lambda df: df.grupo.astype(str)
            )
            .filter([period_column, 'variavel', 'grupo', 'volume',
                     'inadimplencia', 'iv'])
        )

        dfs_features.append(result_df)

    df_feature_metrics = pd.concat(dfs_features, ignore_index=True)

    return df_feature_metrics

def _generate_features_metrics_plots(
    feature_metrics_by_month: pd.DataFrame,
    feature_metrics_by_period: pd.DataFrame,
    features: List[str]) -> Dict:

    feature_metrics_by_month['cohort'] = (
        pd.to_datetime(feature_metrics_by_month['cohort'], format='%Y%m')
    )
    unique_periods = list(feature_metrics_by_period['periodo'].unique())
    len_periods = len(feature_metrics_by_period['periodo'].unique())
    fig = sp.make_subplots(
        rows=3, cols=len_periods,
        specs=[[{"colspan": len_periods}]+[None]*(len_periods-1),
               [{"colspan": len_periods}]+[None]*(len_periods-1),
               [{"secondary_y": True}] * len_periods],
        subplot_titles=(
            ["IV da variável", "Inadimplência por Grupo",] +
            unique_periods
        ),
        vertical_spacing=0.1
    )
    final_trace_number_by_feat = {}
    for feature in features:
        feature_by_month_data = feature_metrics_by_month[feature_metrics_by_month['variavel'] == feature].reset_index(drop=True)
        feature_by_period_data = feature_metrics_by_period[feature_metrics_by_period['variavel'] == feature].reset_index(drop=True)

        df_iv_safra = feature_by_month_data.groupby('cohort').iv.sum().reset_index()
        fig.add_trace(
            go.Scatter(x=df_iv_safra['cohort'], y=df_iv_safra['iv'],
                       mode='lines+markers', name='IV', legendgroup='1'),
            row=1, col=1
        )
        fig.add_shape(type='line',
                  x0=min(df_iv_safra['cohort']), x1=max(df_iv_safra['cohort']),
                  y0=0.1, y1=0.1,
                  line=dict(color='yellow', width=2),
                  row=1, col=1)

        fig.add_shape(type='line',
                      x0=min(df_iv_safra['cohort']), x1=max(df_iv_safra['cohort']),
                      y0=0.3, y1=0.3,
                      line=dict(color='green', width=2),
                      row=1, col=1)

        # Parte de cima: Gráfico de linha para inadimplência por rating por mês
        for group in feature_by_month_data['grupo'].unique():
            group_data = feature_by_month_data[feature_by_month_data['grupo'] == group]
            fig.add_trace(
                go.Scatter(x=group_data['cohort'], y=group_data['inadimplencia'] * 100,
                        mode='lines+markers', name=f'{group}', legendgroup='2'),
                row=2, col=1
            )

        # Parte de baixo: Gráficos para cada período
        unique_periods = feature_by_period_data['periodo'].unique()

        for i, period in enumerate(unique_periods):
            period_data = (
                feature_by_period_data[feature_by_period_data['periodo'] == period]
                .assign(
                    volume_acumulado = lambda df: df.volume.cumsum()
                )
            )
            # Adicionar gráfico de barras para inadimplência
            fig.add_trace(
                go.Bar(x=period_data['grupo'], y=period_data['inadimplencia'] * 100,
                    name='Inadimplência',
                    marker=dict(color='blue'),
                    showlegend=(i == 2), legendgroup='3'),
                row=3, col=int(i+1), secondary_y=False
            )

            # Adicionar gráfico de linha para volume
            fig.add_trace(
                go.Scatter(x=period_data['grupo'], y=period_data['volume_acumulado'] * 100,
                        name='Volume Acumulado',
                        mode='lines+markers',
                        line=dict(color='red'),
                        showlegend=(i == 2), legendgroup='3'),
                row=3, col=int(i+1), secondary_y=True
            )
        final_trace_number_by_feat[feature] = len(fig.data)
    fig.update_xaxes(title_text="Mês", row=1, col=1)
    fig.update_yaxes(title_text="IV", row=1, col=1)
    fig.update_xaxes(title_text="Mês", row=2, col=1)
    fig.update_yaxes(title_text="Inadimplência (%)", row=2, col=1)
    for i, _ in enumerate(unique_periods):
        fig.update_xaxes(title_text="Rating", row=3, col=int(i+1))
        fig.update_yaxes(title_text="Inadimplência (%)", row=3, col=int(i+1), secondary_y=False)
        fig.update_yaxes(title_text="Volume Acumulado (%)", row=3, col=int(i+1), secondary_y=True)

    buttons = [
        dict(
            label="Selecionar uma variável".upper(),
            method="update",
            args=[{"visible": [True] * len(fig.data)},
                  {'title': 'Monitoramento de Inadimplência e Volume: '}]  # Update title
        )
    ]
    for i, feature in enumerate(features):
        button = dict(
            label=f"{feature}",
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {'title': 'Monitoramento de Inadimplência e Volume: '}]  # Update title
        )
        initial_range = final_trace_number_by_feat[features[i-1]] if i > 0 else 0
        final_range = final_trace_number_by_feat[feature]
        for j in range(initial_range,final_range):
            button['args'][0]['visible'][j] = True
        buttons.append(button)
    # Atualizar layout
    fig.update_layout(
        title_text='Monitoramento de Inadimplência e Volume: ',
        height=1200,
        width=1200,
        # showlegend=True,
        legend_tracegroupgap = 350,
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                pad= {"r": 10, "t": 10},
                x=0.345,
                xanchor="left",
                y=1.082,
                yanchor="top"
            )
        ]
    )

    # dict_figures[f"{feature}.html"] = fig

    return fig

def _generate_iv_plots(
    feature_metrics: pd.DataFrame,
    features: List[str]) -> go.Figure:
    # Verifica se o DataFrame contém os dados necessários
    if not {'cohort', 'variavel', 'iv'}.issubset(feature_metrics.columns):
        raise ValueError("O DataFrame deve conter as colunas 'cohort', 'variavel' e 'iv'.")

    feature_metrics['cohort'] = pd.to_datetime(feature_metrics['cohort'], format='%Y%m')

    len_feats = len(features)
    fig = sp.make_subplots(
        rows=len_feats//5+1, cols=5,
        subplot_titles=[f"{var}" for var in features],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    # Adicionar os gráficos de linha para cada variável
    for i, var in enumerate(features):

        row = i // 5 + 1  # Calcula a linha
        col = i % 5 + 1   # Calcula a coluna

        # Filtra o DataFrame para a variável atual
        df_var = feature_metrics[feature_metrics['variavel'] == var]
        df_var = df_var.groupby('cohort')['iv'].sum().reset_index()

        color = SOURCE_COLOR_MAP[var[:3]]
        # Gráfico de linha
        fig.add_trace(
            go.Scatter(x=df_var['cohort'], y=df_var['iv'],
                       mode='lines+markers', name=var, line=dict(color=color),
                       showlegend=False),
            row=row, col=col
        )

        # Adiciona linhas horizontais
        fig.add_shape(type='line',
                      x0=min(df_var['cohort']), x1=max(df_var['cohort']),
                      y0=0.1, y1=0.1,
                      line=dict(color='yellow', width=2),
                      row=row, col=col)

        fig.add_shape(type='line',
                      x0=min(df_var['cohort']), x1=max(df_var['cohort']),
                      y0=0.3, y1=0.3,
                      line=dict(color='green', width=2),
                      row=row, col=col)

        # Atualiza os eixos e o título do subplot
        fig.update_xaxes(title_text="Mês", row=row, col=col)
        fig.update_yaxes(title_text="IV", row=row, col=col)

    # Atualizar o layout
    fig.update_layout(
        title_text='Monitoramento do IV - Mensal',
        height=1000,
        width=2000,
        showlegend=False
    )

    return fig

def performance_monitoring(
    scored_history: pd.DataFrame,
    target_history: pd.DataFrame,
    features:  List[str],
    id_col: str,
    cohort_col: str,
    target_name: str,
    start_cohort: int,
    split_cohort: int,
    final_cohort: int,
) -> np.array:

    id_columns = [id_col,cohort_col]
    df = (
        scored_history.filter(id_columns + features + ['proba', 'score', 'rating'])
        .merge(target_history,
               on=id_columns, how='inner')
        .assign(
            periodo = lambda df: pd.Categorical(_classify_period(df.cohort,
                                                                 start_cohort,
                                                                 split_cohort,
                                                                 final_cohort),
                                                categories=['Treino', 'Teste', 'Produção'],
                                                ordered=True)
        )
    )

    metrics_by_month = _calculate_metrics_by_period(df, 'proba',
                                                    target_name, 'cohort')
    metrics_by_period = _calculate_metrics_by_period(df, 'proba',
                                                    target_name, 'periodo')
    metrics_plot = _generate_metrics_plots(metrics_by_month, metrics_by_period)

    rating_metrics_by_month = _calculate_rating_metrics(df, 'rating',
                                                         target_name, 'cohort')
    rating_metrics_by_period = _calculate_rating_metrics(df, 'rating',
                                                         target_name, 'periodo')
    rating_metrics_plot = _generate_rating_metrics_plots(rating_metrics_by_month,
                                                         rating_metrics_by_period)

    feature_metrics_by_month = _calculate_metrics_per_feature(df, features,
                                                              target_name, 'cohort')
    feature_metrics_by_period = _calculate_metrics_per_feature(df, features,
                                                              target_name, 'periodo')
    feature_metrics_plot = _generate_features_metrics_plots(feature_metrics_by_month,
                                                              feature_metrics_by_period,
                                                              features)
    features_iv_plot = _generate_iv_plots(feature_metrics_by_month, features)

    return (
        metrics_by_month, rating_metrics_by_month,
        metrics_by_period, rating_metrics_by_period,
        metrics_plot, rating_metrics_plot,
        feature_metrics_by_month, feature_metrics_by_period,
        feature_metrics_plot, features_iv_plot
    )

#########################################################################################
################################# INPUT MONITORING ######################################
#########################################################################################


def _calculate_customers_volume(
    df: pd.DataFrame,
    period_column: str
) -> pd.DataFrame:
    
    # Calcular o volume total por safra e por rating
    volume_df = (
        df
        .groupby([period_column]).size().
        reset_index(name='VOLUME_TOTAL')
    )

    return volume_df

def _generate_customers_volume_plot(df_volume: pd.DataFrame) -> go.Figure:

    df_volume['cohort'] = pd.to_datetime(df_volume['cohort'], format='%Y%m')

    # Criar subplots: 2 linhas e 1 coluna
    fig = sp.make_subplots(
        rows=1, cols=1,
        vertical_spacing=0.1
    )

    # Gráfico de linha
    fig.add_trace(
        go.Scatter(x=df_volume['cohort'], y=df_volume['VOLUME_TOTAL'], 
                   mode='lines+markers', name='Volume de clientes', line=dict(color='red'), 
                   showlegend=True),
        row=1, col=1
    )

    fig.update_xaxes(title_text="Mês", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=1, col=1, range=[0,None])

    # Atualizar o layout
    fig.update_layout(
        title_text='Monitoramento do Volume de Clientes Escorados',
        height=600,
        showlegend=True
    )

    return fig

def _calculate_ratings_volume(
    df: pd.DataFrame,
    rating_column: str,
    period_column: str
) -> pd.DataFrame:

    # Calcular o volume total por safra e por rating
    volume_by_rating = (
        df
        .groupby([period_column, rating_column], observed=True)
        .size()
        .reset_index(name='VOLUME_TOTAL')
    )

    # Calcular o volume total por safra
    total_volume_by_safra = (
        df.groupby(period_column).size()
        .reset_index(name=f'VOLUME_{period_column}')
    )

    # Juntar os dois DataFrames
    volume_summary = (
        pd.merge(volume_by_rating, total_volume_by_safra, on=period_column)
        .assign(
            VOLUME_PERCENTUAL = lambda df: df.VOLUME_TOTAL / df[f'VOLUME_{period_column}']
        )
        .filter([period_column, rating_column, 'VOLUME_TOTAL', 'VOLUME_PERCENTUAL'])
    )

    return volume_summary

def _generate_ratings_volume_plot(
    ratings_volume_by_month: pd.DataFrame
) -> go.Figure:

    # Certifique-se de que a coluna cohort está no formato datetime
    ratings_volume_by_month['cohort'] = (
        pd.to_datetime(ratings_volume_by_month['cohort'], format='%Y%m')
    )

    ratings_volume_by_month = (
        ratings_volume_by_month
        .sort_values(['cohort', 'rating'], ascending=[True,False])
        .reset_index(drop=True)
    )

    ratings_volume_by_month = (
        ratings_volume_by_month
        .assign(
            volume_total_stacked = lambda df: df.groupby('cohort').VOLUME_TOTAL.cumsum(),
            volume_percentual_stacked = lambda df: (
                df.groupby('cohort').VOLUME_PERCENTUAL.cumsum() * 100
            )
        )
    )

    # Obter cores distintas para cada rating
    ratings = sorted(ratings_volume_by_month['rating'].unique())
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'cyan', 'magenta']  # Adicione mais cores se necessário

    # Criar subplots: 2 linhas e 1 coluna
    fig = sp.make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.1,
        subplot_titles=("Volume Total por Rating", "Volume Percentual por Rating")
    )

    # Gráfico de linha empilhado para VOLUME_TOTAL
    for i, rating in enumerate(ratings):
        rating_data = ratings_volume_by_month[ratings_volume_by_month['rating'] == rating]
        fig.add_trace(
            go.Scatter(x=rating_data['cohort'], 
                       y=rating_data['volume_total_stacked'], 
                       mode='lines+markers', 
                       name=rating,
                       line=dict(color=colors[i]),
                       fill='tonexty',
                       legendgroup='1'),  # Preenchimento até a linha anterior
            row=1, col=1
        )

    # Gráfico de linha empilhado para VOLUME_PERCENTUAL
    for i, rating in enumerate(ratings):
        rating_data = ratings_volume_by_month[ratings_volume_by_month['rating'] == rating]
        fig.add_trace(
            go.Scatter(x=rating_data['cohort'], 
                       y=rating_data['volume_percentual_stacked'], 
                       mode='lines+markers', 
                       name=f'{rating} (%)',
                       line=dict(color=colors[i], dash='dash'),
                       fill='tonexty',
                       legendgroup='2'),  # Preenchimento até a linha anterior
            row=2, col=1
        )

    # Atualizar os eixos
    fig.update_xaxes(title_text="Mês", row=1, col=1)
    fig.update_xaxes(title_text="Mês", row=2, col=1)
    fig.update_yaxes(title_text="Volume Total", row=1, col=1)
    fig.update_yaxes(title_text="Volume Percentual (%)", row=2, col=1, range=[0, 100])

    # Atualizar o layout
    fig.update_layout(
        title_text='Monitoramento do Volume de Ratings',
        height=1000,
        showlegend=True,
        legend_tracegroupgap = 350,
    )

    return fig

def _generate_migration_matrix_plot(
    df: pd.DataFrame,
    id_column: str,
    rating_column: str,
    period_column: str
) -> plt.Figure:

    df = df.sort_values(period_column).reset_index(drop=True)
    periods = sorted(df[period_column].unique())
    old_period, new_period = periods[-2], periods[-1]

    merged = pd.merge(
        df[df[period_column] == old_period][[id_column, rating_column]],
        df[df[period_column] == new_period][[id_column, rating_column]],
        on=[id_column], suffixes=('_v1', '_v2')
    )

    # Criar a matriz de migração
    migration_matrix = pd.crosstab(merged[f'{rating_column}_v1'], merged[f'{rating_column}_v2']).sort_index(ascending=True)
    migration_matrix = migration_matrix[migration_matrix.columns.sort_values(ascending=False)]

    # Calcular percentuais por linha
    migration_percent = (
        migration_matrix.div(migration_matrix.sum(axis=1), axis=0) * 100
    )

    plot_title = f"Matriz de Migração - {old_period} para {new_period}"
    x_label = f"rating ({new_period})"
    y_label = f"rating ({old_period})"
    annotation_text = (
        migration_matrix.astype(str) + " (" +
        migration_percent.round(1).astype(str) + "%)"
    )
    fig = sp.make_subplots(rows=2,cols=3,
                           specs=[[{"colspan": 3}, None, None],
                                  [{"type": "indicator"},
                                   {"type": "indicator"},
                                   {"type": "indicator"}]],)
    fig.add_trace(
        go.Heatmap(z=migration_percent.values,
                   x=migration_percent.columns,
                   y=migration_percent.index,
                   colorscale='YlGnBu',
                   colorbar=dict(title='Percentual (%)',
                                 y=.8,
                                 len=.5),
                #    coloraxis = "coloraxis1",
                #    colorbar=dict()
                   ),
        row=1, col=1
    )
    fig.update_traces(text=annotation_text, texttemplate="\n%{text}")
    fig.update(layout_coloraxis_showscale=True)

    total_vol = migration_matrix.sum().sum()
    unchanged = np.diag(np.fliplr(migration_matrix.values)).sum() / total_vol
    changed_1 = np.diag(np.fliplr(migration_matrix.values),k=1).sum() / total_vol
    changed_1 = changed_1 + np.diag(np.fliplr(migration_matrix.values),k=-1).sum() / total_vol
    changed_2 = np.diag(np.fliplr(migration_matrix.values),k=2).sum() / total_vol
    changed_2 = changed_2 + np.diag(np.fliplr(migration_matrix.values),k=-2).sum() / total_vol
    fig.add_trace(
        go.Indicator(mode = "number", value = unchanged,
                     title={"text": "Sem mudança"},
                     number=dict(valueformat=".1%")),
        row=2, col=1
    )
    fig.add_trace(
        go.Indicator(mode = "number", value = changed_1,
                     title={"text": "Mudou 1 rating"},
                     number=dict(valueformat=".1%")),
        row=2, col=2
    )
    fig.add_trace(
        go.Indicator(mode = "number", value = changed_2,
                     title={"text": "Mudou 2 ratings"},
                     number=dict(valueformat=".1%")),
        row=2, col=3
    )
    fig.update_layout(title_text=plot_title, title_x=0.5,
                      height=660,width=650,
                      xaxis1=dict(title=x_label),
                      yaxis1=dict(title=y_label)
    )

    return fig

def _calculate_missings_by_feature(
    df: pd.DataFrame,
    period_column: str,
    features: List[str], 
) -> pd.DataFrame:
    
    statistics = []

    grouped = df.groupby(period_column)

    for feature in features:

        for period, group in grouped:
            null_percentage = group[feature].isnull().mean()

            # Adicionar as estatísticas em um dicionário
            statistics.append({
                'cohort': period,
                'variavel': feature,
                'NULOS': null_percentage
            })
    missings_df = pd.DataFrame(statistics)

    return missings_df

def _calculate_volumes_features_groups(
    df: pd.DataFrame,
    features: List[str],
    period_column: str,
    num_bins: int = 5,
    reference_period: str = 'Teste'
) -> pd.DataFrame:

    total_volume_by_safra = (
        df.groupby(period_column).size()
        .reset_index(name=f'VOLUME_{period_column}')
    )

    dfs_features = []

    period_data = df[df['periodo'] == reference_period]
    for feature in features:
        df['grupo'] = _calculate_bins_per_feature(df, feature, period_data, num_bins=num_bins)

        metrics_df = (
            df
            .groupby([period_column, 'grupo'], observed=True   )
            .agg(
                VOLUME_TOTAL=('seller_id', 'size'),
            )
            .reset_index()
        )

        metrics_df = metrics_df.merge(total_volume_by_safra, on=period_column)
        metrics_df['VOLUME_PERCENTUAL'] = (metrics_df['VOLUME_TOTAL'] / metrics_df[f'VOLUME_{period_column}'])
        metrics_df['variavel'] = feature

        result_df = (
            metrics_df
            [[period_column, 'variavel', 'grupo', 'VOLUME_TOTAL', 'VOLUME_PERCENTUAL']]
            .assign(
                grupo = lambda df: df.grupo.astype(str)
            )
        )
        dfs_features.append(result_df)

    df_group_feature_volumes = (
        pd.concat(dfs_features, ignore_index=True)
    )

    return df_group_feature_volumes

def calculate_psi(expected, actual):
    """Calcula o PSI entre duas distribuições."""
    # Adiciona uma pequena constante para evitar divisão por zero e log(0)
    expected = np.where(expected == 0, 1e-10, expected)
    actual = np.where(actual == 0, 1e-10, actual)

    # Calcula o PSI
    psi = np.sum((actual - expected) * np.log(actual / expected))
    return psi

def _calculate_psi_by_feature(
    df: pd.DataFrame, 
    period_column: str, 
    features: List[str],
    split_cohort: int
) -> pd.DataFrame:

    period_data = df[df[period_column] < split_cohort]
    n_samples_reference = period_data[period_data['variavel'] == features[0]].VOLUME_TOTAL.sum()

    psi_results = []

    for feature in features:
        for period in df[period_column].unique():
            expected_counts = (
                period_data[period_data['variavel'] == feature]
                .groupby('grupo').VOLUME_TOTAL.sum()
            ) / n_samples_reference

            actual_counts = (
                df[(df[period_column] == period) & 
                   (df['variavel'] == feature)]
                   [['grupo', 'VOLUME_PERCENTUAL']]
                .set_index('grupo')['VOLUME_PERCENTUAL']
            )

            combined_counts = pd.concat([expected_counts, actual_counts], axis=1).fillna(0)
            combined_counts.columns = ['expected', 'actual']
            psi_value = calculate_psi(combined_counts['expected'], combined_counts['actual'])
            psi_results.append({
                'cohort': period,
                'variavel': feature,
                'PSI': psi_value
            })

    psi_df = pd.DataFrame(psi_results)

    return psi_df

def _calculate_features_statistics(
    df: pd.DataFrame,
    period_column: str,
    features: List[str],
) -> pd.DataFrame:

    statistics = []

    grouped = df.groupby(period_column)

    for feature in features:
        if pd.api.types.is_numeric_dtype(df[feature]):

            for period, group in grouped:
                mean = group[feature].mean()
                minimum = group[feature].min()
                maximum = group[feature].max()
                std_dev = group[feature].std()
                percentiles = np.percentile(group[feature].dropna(), [1, 5, 25, 50, 75, 90, 95, 99])

                # Adicionar as estatísticas em um dicionário
                statistics.append({
                    'cohort': period,
                    'variavel': feature,
                    'MEDIA': mean,
                    'MIN': minimum,
                    'MAX': maximum,
                    'STD': std_dev,
                    'P1': percentiles[0],
                    'P5': percentiles[1],
                    'P25': percentiles[2],
                    'P50': percentiles[3],
                    'P75': percentiles[4],
                    'P90': percentiles[5],
                    'P95': percentiles[6],
                    'P99': percentiles[7]
                })
    stats_df = pd.DataFrame(statistics)

    return stats_df

def _generate_statistics_plot(
    features_statistics: pd.DataFrame,
    features_groups_volume: pd.DataFrame,
    features: List[str]) -> go.Figure:
    # Verifica se o DataFrame contém os dados necessários
    if not {'cohort', 'variavel', 'PSI',
            'shap_importance'}.issubset(features_statistics.columns):
        message = (
            "O DataFrame features_statistics deve conter as colunas 'cohort', 'variavel', 'PSI' e "
            "'shap_importance'."
        )
        raise ValueError(message)

    if not {'cohort', 'grupo',
            'VOLUME_TOTAL',
            'VOLUME_PERCENTUAL'}.issubset(features_groups_volume.columns):
        message = (
            "O DataFrame features_groups_volume deve conter as colunas 'cohort', 'grupo', "
            "'VOLUME_TOTAL' e 'VOLUME_PERCENTUAL'."
        )
        raise ValueError(message)

    features_statistics['cohort'] = pd.to_datetime(features_statistics['cohort'], format='%Y%m')
    features_groups_volume['cohort'] = pd.to_datetime(features_groups_volume['cohort'], format='%Y%m')

    # Criar subplots: 4 linhas e 4 colunas
    fig = sp.make_subplots(
        rows=6, cols=1,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    final_trace_number_by_feat = dict()
    # Adicionar os gráficos de linha para cada variável
    for i, feature in enumerate(features):
        # Filtra o DataFrame para a variável atual
        df_var = features_statistics[features_statistics['variavel'] == feature]

        # Gráfico de linha
        fig.add_trace(
            go.Scatter(x=df_var['cohort'], y=df_var['PSI'],
                       mode='lines+markers', name='PSI', line=dict(),
                       showlegend=True, legendgroup='1'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=df_var['cohort'], y=(1-df_var['NULOS'])*100,
                       mode='lines+markers', name='Preenchimento (%)', line=dict(),
                       showlegend=True, legendgroup='2'),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=df_var['cohort'], y=df_var['MIN'],
                       mode='lines+markers', name='MIN', line=dict(),
                       showlegend=True, legendgroup='3'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_var['cohort'], y=df_var['P25'],
                       mode='lines+markers', name='P25', line=dict(),
                       showlegend=True, legendgroup='3'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_var['cohort'], y=df_var['P50'],
                       mode='lines+markers', name='P50', line=dict(),
                       showlegend=True, legendgroup='3'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_var['cohort'], y=df_var['P75'],
                       mode='lines+markers', name='P75', line=dict(),
                       showlegend=True, legendgroup='3'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_var['cohort'], y=df_var['MAX'],
                       mode='lines+markers', name='MAX', line=dict(),
                       showlegend=True, legendgroup='3'),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(x=df_var['cohort'], y=df_var['shap_importance'],
                       mode='lines+markers', name='shap_importance', line=dict(),
                       showlegend=True, legendgroup='4'),
            row=4, col=1
        )

        feature_by_month_data = (
            features_groups_volume[features_groups_volume['variavel'] == feature]
            .assign(
                volume_total_stacked = lambda df: df.groupby('cohort').VOLUME_TOTAL.cumsum(),
                volume_percentual_stacked = lambda df: (
                    df.groupby('cohort').VOLUME_PERCENTUAL.cumsum() * 100
                )
            )
            .assign(
                key_sort = lambda df: pd.to_numeric(
                    df.grupo.str.split(',').str[0].str.lstrip('('),
                    errors='coerce'
                )
            )
            .sort_values(['cohort','key_sort'], na_position='last')
            .drop(columns=['key_sort'])
            .reset_index(drop=True)
        )
        # Obter cores distintas para cada rating
        groups = list(feature_by_month_data['grupo'].unique())
        # Adicione mais cores abaixo, se necessário
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'cyan', 'magenta']
        # Gráfico de linha empilhado para VOLUME_TOTAL
        for i, group in enumerate(groups):
            rating_data = feature_by_month_data[feature_by_month_data['grupo'] == group]
            fig.add_trace(
                go.Scatter(x=rating_data['cohort'],
                        y=rating_data['volume_total_stacked'],
                        mode='lines+markers',
                        name=group,
                        line=dict(color=colors[i % len(colors)]),
                        fill='tonexty',
                        legendgroup='5'),  # Preenchimento até a linha anterior
                row=5, col=1
            )

        # Gráfico de linha empilhado para VOLUME_PERCENTUAL
        for i, group in enumerate(groups):
            rating_data = feature_by_month_data[feature_by_month_data['grupo'] == group]
            fig.add_trace(
                go.Scatter(x=rating_data['cohort'],
                        y=rating_data['volume_percentual_stacked'],
                        mode='lines+markers',
                        name=f'{group} (%)',
                        line=dict(color=colors[i % len(colors)], dash='dash'),
                        fill='tonexty',
                        legendgroup='6',
                        showlegend=False),  # Preenchimento até a linha anterior
                row=6, col=1
            )

        # Atualiza os eixos e o título do subplot
        final_trace_number_by_feat[feature] = len(fig.data)

    # Adiciona linhas horizontais
    fig.add_shape(type='line',
                  x0=min(features_statistics['cohort']), 
                  x1=max(features_statistics['cohort']),
                  y0=0.1, y1=0.1,
                  line=dict(color='yellow', width=2),
                  row=1, col=1)

    fig.add_shape(type='line',
                  x0=min(features_statistics['cohort']),
                  x1=max(features_statistics['cohort']),
                  y0=0.2, y1=0.2,
                  line=dict(color='red', width=2),
                  row=1, col=1)

    fig.update_xaxes(title_text="Mês", row=1, col=1)
    fig.update_yaxes(title_text="PSI", row=1, col=1)
    fig.update_xaxes(title_text="Mês", row=2, col=1)
    fig.update_yaxes(title_text="Preenchimento (%)", row=2, col=1, range=[0,None])
    fig.update_xaxes(title_text="Mês", row=3, col=1)
    fig.update_yaxes(title_text="Valores", row=3, col=1)
    fig.update_xaxes(title_text="Mês", row=4, col=1)
    fig.update_yaxes(title_text="Importância SHAP", row=4, col=1)
    fig.update_xaxes(title_text="Mês", row=5, col=1)
    fig.update_yaxes(title_text="Volume Total", row=5, col=1)
    fig.update_xaxes(title_text="Mês", row=6, col=1)
    fig.update_yaxes(title_text="Volume Percentual (%)", row=6, col=1, range=[0, 100])


    buttons = [
        dict(
            label="Selecionar uma variável".upper(),
            method="update",
            args=[{"visible": [True] * len(fig.data)},  # Update visibility
                  {"title": 'Estatísticas mensais: '},]  # Update title
        )
    ]
    for i, feature in enumerate(features):
        button = dict(
            label=f"{feature}",
            method="update",
            args=[{"visible": [False] * len(fig.data)},  # Update visibility
                  {"title": 'Estatísticas mensais: '}]  # Update title
        )
        initial_range = final_trace_number_by_feat[features[i-1]] if i > 0 else 0
        final_range = final_trace_number_by_feat[feature]
        for j in range(initial_range,final_range):
            button['args'][0]['visible'][j] = True
        buttons.append(button)

    # Atualizar o layout
    fig.update_layout(
        title_text='Estatísticas mensais: ',
        height=1500,
        width=1000,
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                pad= {"r": 10, "t": 10},
                x=0.20,
                xanchor="left",
                y=1.06,
                yanchor="top"
            )
        ],
        legend_tracegroupgap = 210,
    )

    return fig

def _generate_psi_plots(
    features_statistics: pd.DataFrame,
    features: List[str]) -> go.Figure:
    # Verifica se o DataFrame contém os dados necessários
    if not {'cohort', 'variavel', 'PSI'}.issubset(features_statistics.columns):
        raise ValueError("O DataFrame deve conter as colunas 'cohort', 'variavel' e 'PSI'.")

    features_statistics['cohort'] = pd.to_datetime(features_statistics['cohort'], format='%Y%m')

    len_feats = len(features)
    fig = sp.make_subplots(
        rows=len_feats//5+1, cols=5,
        subplot_titles=[f"{var}" for var in features],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    # Adicionar os gráficos de linha para cada variável
    for i, var in enumerate(features):

        row = i // 5 + 1  # Calcula a linha
        col = i % 5 + 1   # Calcula a coluna

        # Filtra o DataFrame para a variável atual
        df_var = features_statistics[features_statistics['variavel'] == var]

        # Gráfico de linha
        color = SOURCE_COLOR_MAP[var[:3]]
        fig.add_trace(
            go.Scatter(x=df_var['cohort'], y=df_var['PSI'],
                       mode='lines+markers', name=var, line=dict(color=color),
                       showlegend=False),
            row=row, col=col
        )

        # Adiciona linhas horizontais
        fig.add_shape(type='line',
                      x0=min(df_var['cohort']), x1=max(df_var['cohort']),
                      y0=0.1, y1=0.1,
                      line=dict(color='yellow', width=2),
                      row=row, col=col)

        fig.add_shape(type='line',
                      x0=min(df_var['cohort']), x1=max(df_var['cohort']),
                      y0=0.2, y1=0.2,
                      line=dict(color='red', width=2),
                      row=row, col=col)

        # Atualiza os eixos e o título do subplot
        fig.update_xaxes(title_text="Mês", row=row, col=col)
        fig.update_yaxes(title_text="PSI", row=row, col=col)

    # Atualizar o layout
    fig.update_layout(
        title_text='Monitoramento do PSI - Mensal',
        height=1000,
        width=2000,
        showlegend=False
    )

    return fig

def prediction_monitoring(
    scored_history: pd.DataFrame,
    df_shap: pd.DataFrame,
    features: List[str],
    id_col: str,
    cohort_col: str,
    start_cohort: int,
    split_cohort: int,
    final_cohort: int,
) -> Dict:
    """

    """
    features = [feat for feat in features if feat != 'sel_seller_state']
    id_columns = [id_col, cohort_col]
    df = (
        scored_history.filter(id_columns + features + [
            'probabilidade_inadimplencia', 'score', 'rating'
        ])
        .assign(
            periodo = lambda df: pd.Categorical(_classify_period(df.cohort,
                                                  start_cohort, split_cohort, final_cohort),
                                                categories=['Treino', 'Teste', 'Produção'],
                                                ordered=True)
        )
    )

    customers_volume_by_month = _calculate_customers_volume(df, 'cohort')
    customers_volume_by_month_plot = (
        _generate_customers_volume_plot(customers_volume_by_month.copy())
    )

    ratings_volume_by_month = _calculate_ratings_volume(df, 'rating', 'cohort')
    ratings_volume_by_month_plot = _generate_ratings_volume_plot(ratings_volume_by_month.copy())

    migration_matrix_plot = _generate_migration_matrix_plot(df.copy(), 'seller_id',
                                                            'rating', 'cohort')

    missings_by_feature_df = _calculate_missings_by_feature(df, 'cohort',features)
    features_groups_volume = _calculate_volumes_features_groups(df, features,
                                                                'cohort', num_bins = 5)
    psi_by_feature_df = _calculate_psi_by_feature(features_groups_volume, 'cohort',
                                                  features, final_cohort)
    features_statistics = _calculate_features_statistics(df, 'cohort',
                                                         features)
    df_shap_safra = (
        df_shap
        .filter(['cohort']+features)
        .abs()
        .groupby(['cohort'], as_index=False)
        .mean()
        .melt(id_vars='cohort',value_vars=features,
              var_name='variavel',value_name='shap_importance')
        .assign(
            shap_importance = lambda df: (
                df.shap_importance/df.groupby('cohort').shap_importance.transform('sum')
            )
        )
    )

    features_statistics = (
        features_statistics
        .merge(missings_by_feature_df, on=['cohort', 'variavel'], how='inner')
        .merge(psi_by_feature_df, on=['cohort', 'variavel'], how='inner')
        .merge(df_shap_safra, on=['cohort', 'variavel'], how='inner')
    )

    features_statistics_plot = _generate_statistics_plot(features_statistics.copy(),
                                                         features_groups_volume,
                                                         features)

    features_psi_plot = _generate_psi_plots(features_statistics, features)

    return [
        customers_volume_by_month,
        customers_volume_by_month_plot,
        ratings_volume_by_month,
        ratings_volume_by_month_plot,
        migration_matrix_plot,
        features_groups_volume,
        features_statistics,
        features_statistics_plot,
        features_psi_plot
    ]

def _generate_shap_values_plot(
    df_shap: pd.DataFrame,
    df_features: pd.DataFrame,
    features: List[str],
    period_column: str
) -> plt.Figure:

    """
    Gera uma figura do matplotlib com subplots de SHAP summary_plot e gráficos de importância relativa por período.

    Parâmetros:
    - df: DataFrame contendo as variáveis originais e a coluna 'periodo'.
    - shap_values: DataFrame ou array com os valores SHAP correspondentes a `df`.

    Retorna:
    - fig: Objeto da figura gerada.
    """

    # Identificar períodos únicos na coluna 'periodo'
    unique_periods = df_shap[period_column].unique()
    n_periods = len(unique_periods)

    n_rows = 1

    df_features_sampled = (
        df_features.loc[df_shap.index,:]
        .assign(**{period_column: df_shap[period_column]})
    )

    if any(df_shap[period_column].reset_index(drop=True)!=
           df_features_sampled[period_column].reset_index(drop=True)):
        message = (
            f"Different column: {period_column}"
        )
        raise ValueError(message)

    # Criar figura
    fig = plt.figure(figsize=(int(8 * n_periods), 6))
    for i, period in enumerate(unique_periods):

        # Filtrar os dados para o período atual
        period_data = df_features_sampled[df_features_sampled[period_column] == period]
        X_period = period_data[features]  # Subset de X
        period_shap_data = df_shap[df_shap[period_column] == period]
        shap_period = period_shap_data[features]  # Subset dos valores SHAP

        ax = plt.subplot(n_rows, n_periods, i+1)
        shap.summary_plot(shap_period.values, X_period, show=False, title='', plot_size=None)
        ax.set_title(f'Valores SHAP ({period})')
        ax.set_xlabel('')

    # Ajustar o layout para evitar sobreposição
    plt.tight_layout()

    return fig

def _generate_shap_feature_importance_plot(
    df_shap: pd.DataFrame,
    features: List[str],
    period_column: str
) -> plt.Figure:

    # Identificar períodos únicos na coluna 'periodo'
    unique_periods = df_shap[period_column].unique()
    n_periods = len(unique_periods)

    # Criar figura
    subplot_titles = [f'Importância das Features (%) - ({period})' for period in unique_periods]
    plotly_fig = sp.make_subplots(
        rows=n_periods, cols=1,
        vertical_spacing=0.1,
        subplot_titles=subplot_titles
    )
    for i, period in enumerate(unique_periods):

        # Filtrar os dados para o período atual
        period_data = df_shap[df_shap[period_column] == period]
        shap_period = period_data[features]  # Subset dos valores SHAP

        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': np.abs(shap_period).mean(axis=0)
        }).sort_values(by='Importance', ascending=False)
        feature_importance['Normalized Importance'] = (
            feature_importance['Importance'] / feature_importance['Importance'].sum()
        )
        feature_importance['cumsum'] = feature_importance['Normalized Importance'].cumsum()
        feature_importance['Fonte'] = feature_importance['Feature'].apply(lambda x: x[:3])
        feature_importance.set_index('Feature', inplace=True)
        feature_importance_sorted = feature_importance.iloc[::-1]
        cores = feature_importance_sorted['Fonte'].map(SOURCE_COLOR_MAP)
        plotly_fig.add_trace(
            go.Bar(x=feature_importance_sorted['Normalized Importance']*100,
                   y=feature_importance_sorted.index,
                   marker=dict(color=list(cores)),
                   orientation='h'),
            row=i+1, col=1
        )

    plotly_fig.update_layout(height=800,width=750,showlegend=False,)
    plt.tight_layout()

    return plotly_fig

def generate_shap_values(
    scored_history: pd.DataFrame,
    trained_model: CustomLGBMClassifier,
    id_col: str,
    cohort_col: str,
    start_cohort: int,
    split_cohort: int,
    final_cohort: int,
    sample_size: int,
    random_state: int,
):
    """
    
    """

    id_columns = [id_col, cohort_col]
    features = list(trained_model.feature_names_in_)
    df = (
        scored_history.filter(id_columns + features + [
            'probabilidade_inadimplencia', 'score', 'rating'
        ])
        .assign(
            periodo = lambda df: pd.Categorical(
                _classify_period(df.cohort, start_cohort, split_cohort, final_cohort),
                categories=['Treino', 'Teste', 'Produção'],
                ordered=True
            )
        )
    )

    df_period_list = []
    for period in df.periodo.unique():
        df_period = df[df.periodo==period]
        if len(df_period) > sample_size:
            df_period = df_period.sample(sample_size,random_state=random_state) 
        df_period_list.append(df_period)
    df = pd.concat(df_period_list,axis=0)


    explainer = shap.TreeExplainer(trained_model.classifier[-1])
    features = trained_model.feature_names_in_
    X_transf = df[features]
    shap_values = explainer.shap_values(X_transf)

    df_shap_values = (
        pd.DataFrame(shap_values, columns=features, index=df.index)
        .assign(
            seller_id = df.seller_id,
            cohort = df.cohort,
            periodo = df.periodo,
        )
    )

    return df_shap_values

def shap_monitoring(
    df_shap_values: pd.DataFrame,
    score_history: pd.DataFrame,
    features: List[str]
):
    """
    
    """

    shap_summary_plot = _generate_shap_values_plot(df_shap_values,
                                                   score_history,
                                                   features,
                                                   'periodo')
    shap_ft_imp_plot = _generate_shap_feature_importance_plot(df_shap_values,
                                                              features,
                                                              'periodo')

    return (
        shap_summary_plot,
        shap_ft_imp_plot
    )
