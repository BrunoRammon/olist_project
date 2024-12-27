from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
import shap
import pandas as pd
from sklearn.pipeline import Pipeline
import string
from tabulate import tabulate
import plotly.graph_objects as go
from scipy.stats import ks_2samp
from plotly.subplots import make_subplots

def _configure_axes(ax, title_names, secundary_color, y_inf_lim=0.0,
                   turn_off_legend=True, xlabelrotation=0.0,
                   turn_off_yspines={'left': False, 'right': True}, turn_off_yticks=False,
                   turn_off_xspines={'bottom': False, 'top': True},
                   format_str = '%.2f', size='small'):
    ax.grid(False)
    ax.spines['bottom'].set_color(secundary_color)
    ax.spines['left'].set_color(secundary_color)
    ax.spines['right'].set_color(secundary_color)
    ax.spines['top'].set_color(secundary_color)
    if turn_off_yspines.get('left', None):
        ax.spines['left'].set_visible(False)
    if turn_off_yspines.get('right', None):
        ax.spines['right'].set_visible(False)
    if turn_off_xspines.get('top', None):
        ax.spines['top'].set_visible(False)
    if turn_off_xspines.get('bottom', None):
        ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='x',colors=secundary_color,labelsize=size,
                   which='both',labelrotation=xlabelrotation)
    ax.tick_params(axis='y', length=0, width=0, colors=secundary_color,
                   labelsize=size,which='both')
    if turn_off_yticks:
        ax.tick_params(axis='y', length=0, width=0, colors=secundary_color,
                       labelsize=size,which='both', left=False, right=False,
                       labelleft=False, labelright=False)

    if turn_off_legend:
        try:
            ax.get_legend().remove()
        except:
            pass

    # axis set up
    ax.set_ylim(bottom=y_inf_lim)
    ax.set_ylabel(title_names['ytitle'].upper(), loc='top',
                    fontsize=size, color=secundary_color)
    ax.set_xlabel(title_names['xtitle'].upper(), loc='left',
                    fontsize=size, color=secundary_color)
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter(format_str))

    # title
    ax.set_title(label=title_names['title'], loc='left', x=0, y=1.0,
                    color=secundary_color)
    return ax

def _annotations_over_bars_chart(ax, secundary_color, format_str='.2f', size='small'):
    # plot annotation in bar charts
    for bar in ax.patches:
        ax.annotate(format(bar.get_height(), format_str),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='center',
                    fontsize=size, xytext=(0, 8),
                    color=secundary_color,
                    textcoords='offset points')
    return ax

def _annotations_in_line_points(ax, color, ha='right', va='bottom'):
    # plot annotation in bar charts
    line = ax.get_lines()[-1]
    y_data = line.get_ydata()
    x_data = line.get_xdata()
    for x,y in zip(x_data, y_data):
        ax.annotate(f'{y:2.1f}%', (x, y), ha=ha, va=va,
                    fontsize='small', color=color)
    return ax

def _set_same_y_sup_lim(ax_1,ax_2):
    _, y_sup_1 = ax_1.get_ylim()
    _, y_sup_2 = ax_2.get_ylim()
    y_sup_max = y_sup_1 if y_sup_1 > y_sup_2 else y_sup_2
    ax_1.set_ylim(top=y_sup_max)
    ax_2.set_ylim(top=y_sup_max)

def _set_same_y_sup_lim_2(*axes):
    y_sup_list = [ax.get_ylim()[1] for ax in axes]
    y_sup_max = max(y_sup_list)
    for ax in axes:
        ax.set_ylim(top=y_sup_max)

def _annotate_oot_division(division_date, ax, stressed_color):
    ax.axvline(x=division_date, color=stressed_color,
                 linestyle='--')
    y_inf, _ = ax.get_ylim()
    ax.annotate('OOT'.upper(),xy=(division_date,y_inf+.001),
                  fontsize='small', color=stressed_color, ha='left',
                  va='bottom', weight='semibold')
    ax.annotate('Treino/OOS'.upper(),xy=(division_date,y_inf+.001),
                  fontsize='small', color=stressed_color, ha='right',
                  va='bottom', weight='semibold')


def _change_date_format_xlabels_bar_time_series(ax):
    xtl=[item.get_text()[:10] for item in ax.get_xticklabels()]
    xtl=['-'.join(item.split('-')[-2::-1])  for item in xtl]
    _=ax.set_xticklabels(xtl)

def _configure_xaxes_line_time_series(ax):
    x_data = ax.get_lines()[0].get_xdata()
    min_x = x_data.min()
    max_x = x_data.max()
    for line in ax.get_lines()[1:]:
        x_data = line.get_xdata()
        min_x = x_data.min() if min_x > x_data.min() else min_x
        max_x = x_data.max() if max_x < x_data.max() else max_x
    ax.set_xlim(left=min_x,right=max_x)
    from matplotlib.dates import DateFormatter
    date_format = DateFormatter('%m-%Y')
    ax.xaxis.set_major_formatter(date_format)

def _annotate_label_time_series(ax,color):
    # annotating series label
    for line in ax.get_lines():
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        max_x_index = np.argmax(x_data)
        max_x = x_data[max_x_index]
        max_y = y_data[max_x_index]

        ax.annotate(f'{line.get_label()}',xy=(max_x,max_y),
                    xytext=(max_x,max_y), fontsize='small',
                    color=color, ha='left', va='center',
                    weight='semibold')

def _annotate_first_and_last_points_time_series(ax,stressed_color,
                                               secundary_color):
    for line in ax.get_lines():
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        max_x_index = np.argmax(x_data)
        max_x = x_data[max_x_index]
        max_y = y_data[max_x_index]

        ax.annotate(f'{max_y:.2f}',xy=(max_x,max_y),
                    xytext=(max_x,max_y-.05), fontsize='small',
                    color=stressed_color, ha='left', va='center',
                    weight='semibold',
                    arrowprops=dict(color=secundary_color, arrowstyle='->'))

        min_x_index = np.argmin(x_data)
        min_x = x_data[min_x_index]
        min_y = y_data[min_x_index]
        ax.annotate(f'{min_y:.2f}',xy=(min_x,min_y),
                    xytext=(min_x,min_y-.05), fontsize='small',
                    color=stressed_color, ha='left', va='center',
                    weight='semibold',
                    arrowprops=dict(color=secundary_color, arrowstyle='->'))

def _calculate_auc(df_probas):
    auc = roc_auc_score(df_probas['target'],df_probas['probas'])
    return auc

def _roc(y_target, y_probas, ax):
    return skplt.metrics.plot_roc(y_target, y_probas, ax=ax)

def _ks_statistic(y_target, y_probas, ax):
    return skplt.metrics.plot_ks_statistic(y_target, y_probas, ax=ax)

def _precision_recall(y_target, y_probas, ax):
    return skplt.metrics.plot_precision_recall(y_target, y_probas, ax=ax)

def _cumulative_gain(y_target, y_probas, ax):
    return skplt.metrics.plot_cumulative_gain(y_target, y_probas, ax=ax)

def _lift_curve(y_target, y_probas, ax):
    return skplt.metrics.plot_lift_curve(y_target, y_probas, ax=ax)

def calculate_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value

    return expected_value, shap_values

def _beeswarm_from_shap_values(shap_values, X):
    shap.summary_plot(shap_values, X, show=False,
                      title='', plot_size=None)

def _train_test_validation_draw_volumetry_rate_over_time_plot(df_train_probas, df_test_probas, 
                                                             df_val_probas, ax, title_names,
                                                             stressed_color, secundary_color):
    tol_vol = len(df_train_probas)+len(df_test_probas)+len(df_val_probas)
    ax = (
        pd.concat([df_train_probas, df_test_probas, df_val_probas],axis=0)
        .groupby('Mes_M')
        .agg(volumetry=('target','count'))
        .assign(rate_vol = lambda df: df['volumetry']/tol_vol)
        .filter(['rate_vol'])
        .plot.bar(color=stressed_color, ax=ax)
    )
    
    _configure_axes(ax, title_names, secundary_color, xlabelrotation=45)
    _annotations_over_bars_chart(ax, secundary_color)
    _change_date_format_xlabels_bar_time_series(ax)
    return ax

def _train_test_validation_volumetry_rate_over_time_plot(df_train_probas,df_test_probas, 
                                                        df_val_probas,
                                                        stressed_color, secundary_color, 
                                                        save_to_file=None,
                                                        return_fig=False):
    title_names =  {'title':'Taxa de volume ao longo do tempo',
                    'xtitle':'Safras',
                    'ytitle':'Taxa de volume'}
    fig, ax = plt.subplots(figsize=(13,6))
    _train_test_validation_draw_volumetry_rate_over_time_plot(df_train_probas, df_test_probas, 
                                                             df_val_probas, ax, title_names,
                                                             stressed_color, secundary_color)
    if save_to_file:
        plt.savefig(f"{save_to_file}")
    plt.show()
    if return_fig:
        return fig

def _train_test_validation_draw_auc_over_time_plot(df_train_probas, df_test_probas, df_val_probas,
                                                  ax, title_names,
                                                  stressed_color, secundary_color):
    def custom_agg(x):
        try:
            auc = roc_auc_score(x['target'],x['probas'])
        except ValueError:
            auc = -0.5*((x['target']-x['probas']).abs().mean())+1
        return auc

    df_auc_train = (
        df_train_probas
        .groupby('Mes_M')
        .apply(custom_agg)
        .reset_index(name='auc')
    )
    df_auc_test = (
        df_test_probas
        .groupby('Mes_M')
        .apply(custom_agg)
        .reset_index(name='auc')
    )
    df_auc_val = (
        df_val_probas
        .groupby('Mes_M')
        .apply(custom_agg)
        .reset_index(name='auc')
    )
    ax.plot(df_auc_train['Mes_M'], df_auc_train['auc'], color=secundary_color, label='TREINO')
    ax.plot(df_auc_test['Mes_M'], df_auc_test['auc'], color=secundary_color, label='OOS', linestyle='--')
    ax.plot(df_auc_val['Mes_M'], df_auc_val['auc'], color=secundary_color, label='OOT')
    _annotate_label_time_series(ax,stressed_color)

    _configure_xaxes_line_time_series(ax)
    _configure_axes(ax, title_names, secundary_color)
    ax.set_ylim(bottom=0.0, top=1.0)


    _annotate_first_and_last_points_time_series(ax,stressed_color,secundary_color)

    division_date = df_val_probas['Mes_M'].min()
    ax.set_ylim(bottom=0.5,top=1.0)
    _annotate_oot_division(division_date, ax, stressed_color)

def _train_test_validation_auc_over_time(df_train_probas, df_test_probas, df_val_probas,
                                        stressed_color, secundary_color, save_to_file=None,
                                        return_fig=False):
    title_names = {'title':'ROC AUC ao longo do tempo',
                   'xtitle':'Safras',
                   'ytitle':'ROC AUC'}
    fig,ax = plt.subplots(figsize=(13,6))
    _train_test_validation_draw_auc_over_time_plot(df_train_probas, df_test_probas, df_val_probas,
                                                  ax, title_names, 
                                                  stressed_color, secundary_color)
    if save_to_file:
        plt.savefig(f"{save_to_file}")
    plt.show()

    if return_fig:
        return fig

def _train_test_validation_metric_curve(df_train_probas, df_test_probas, df_val_probas,
                                        function, secundary_color, save_to_file=None,
                                        return_fig=False):

    fig = plt.figure(figsize=(20,6))
    data = {'treino':df_train_probas,
            'teste':df_test_probas,
            'validation':df_val_probas}
    for i,(name,df) in enumerate(data.items(), start=1):
        ax = plt.subplot(1,3,i)
        y_probas_0 = 1-df['probas']
        y_probas_1 = df['probas']
        y_probas = np.array([y_probas_0, y_probas_1]).transpose()
        function(df['target'], y_probas, ax=ax)
        title = ax.get_title()
        xtitle = ax.get_xlabel()
        ytitle = ax.get_ylabel()
        ax.set_title('')
        _configure_axes(ax, {'title':f'{title} ({name})',
                            'xtitle':f'{xtitle}',
                            'ytitle':f'{ytitle}'}, secundary_color,
                            turn_off_legend=False)
    if save_to_file:
        plt.savefig(f"{save_to_file}")
    plt.show()

    if return_fig:
        return fig

def _train_test_validation_ordenation_graphs(df_train_probas, df_test_probas, df_val_probas, n_percentiles,
                                            stressed_color,secundary_color, 
                                            save_img_to_file=None, save_rating_limits_to_file=None,
                                            return_fig=False):
    fig, ax = plt.subplots(1,3,figsize=(20,6))

    dict_ratings_limits_all = dict()
    dfs_list = [df_train_probas,df_test_probas,df_val_probas]
    names_list = ['treino','teste','validação']
    for i,(df,name) in enumerate(zip(dfs_list,names_list)):
        df['proba_bins'], bins_lims = pd.qcut(df['probas'], q=n_percentiles, retbins=True, 
                                              duplicates='drop')
        n_unique = df['proba_bins'].nunique()
        rating_names = list(string.ascii_uppercase[:n_unique])
        df['proba_bins'] = df['proba_bins'].cat.rename_categories(rating_names)
        bins_lims[0]=0
        bins_lims[-1]=1
        dict_ratings_limits = dict()
        for j, cat in enumerate(df['proba_bins'].cat.categories):
            # print(f'{cat}: ({bins_lims[j]:.3f},{bins_lims[j+1]:.3f})')
            dict_ratings_limits[cat] = f'({bins_lims[j]:.3f},{bins_lims[j+1]:.3f})'
        dict_ratings_limits_all[name] = dict_ratings_limits
        
        (
            df
            .groupby('proba_bins', observed=True)['target'].mean()
            .plot.bar(color=stressed_color, ax=ax[i])
        )
    df_ratings_limits = pd.DataFrame.from_dict(dict_ratings_limits_all)
    tabular_df = tabulate(df_ratings_limits, headers='keys', tablefmt='psql')
    print(tabular_df)
    

    for i,tp in enumerate(names_list):
        _configure_axes(ax[i], {'title':f'Gráfico de ordenação ({tp})',
                               'xtitle':'Percentis de probabilidade',
                               'ytitle':'Taxa de ruins'},
                            secundary_color=secundary_color)
        _annotations_over_bars_chart(ax[i], secundary_color)

    _set_same_y_sup_lim_2(ax[0],ax[1],ax[2])

    if save_img_to_file:
        plt.savefig(f"{save_img_to_file}")
        if save_rating_limits_to_file:
            with open(f'{save_rating_limits_to_file}', 'w') as f:
                f.write(tabular_df)
    plt.show()

    if return_fig:
        return fig
def train_test_validation_metrics(y_train, y_test, y_val,
                                  y_probas_train, y_probas_test, y_probas_val, 
                                  cohort_train, cohort_test, cohort_val,
                                  n_percentiles=None, save_figures=False, saving_folder='.'):

    df_train_probas = pd.DataFrame()
    df_test_probas = pd.DataFrame()
    df_val_probas = pd.DataFrame()

    df_train_probas['target'] = y_train.reset_index(drop=True)
    df_test_probas['target'] = y_test.reset_index(drop=True)
    df_val_probas['target'] = y_val.reset_index(drop=True)

    df_train_probas['probas'] = y_probas_train
    df_test_probas['probas'] = y_probas_test
    df_val_probas['probas'] = y_probas_val

    df_train_probas['Mes_M'] = cohort_train.reset_index(drop=True)
    df_test_probas['Mes_M'] = cohort_test.reset_index(drop=True)
    df_val_probas['Mes_M'] = cohort_val.reset_index(drop=True)

    stressed_color = '#006e9cff'
    secundary_color = 'gray'

    files_names = []

    print()
    save_to_file = f"{saving_folder}/train_test_validation_volumetry_rate_over_time_plot.png" if save_figures else None
    _train_test_validation_volumetry_rate_over_time_plot(df_train_probas,df_test_probas,df_val_probas,
                                                        stressed_color, secundary_color,
                                                        save_to_file=save_to_file)
    files_names.append(save_to_file)

    print()
    auc_train = _calculate_auc(df_train_probas)
    auc_test = _calculate_auc(df_test_probas)
    auc_val = _calculate_auc(df_val_probas)
    print(f'ROC AUC (treino): {auc_train}')
    print(f'ROC AUC (teste): {auc_test}')
    print(f'ROC AUC (validação): {auc_val}')

    print()
    save_to_file = f"{saving_folder}/train_test_validation_auc_over_time.png" if save_figures else None
    _train_test_validation_auc_over_time(df_train_probas, df_test_probas, df_val_probas,
                                        stressed_color, secundary_color, 
                                        save_to_file=save_to_file)
    files_names.append(save_to_file)

    print()
    save_to_file = f"{saving_folder}/train_test_validation_metric_curve.png" if save_figures else None
    _train_test_validation_metric_curve(df_train_probas, df_test_probas, df_val_probas,
                                       _roc, secundary_color, 
                                       save_to_file=save_to_file)
    files_names.append(save_to_file)

    print()
    save_to_file = f"{saving_folder}/train_test_validation_ks_curve.png" if save_figures else None
    _train_test_validation_metric_curve(df_train_probas, df_test_probas, df_val_probas,
                                       _ks_statistic, secundary_color, 
                                       save_to_file=save_to_file)
    files_names.append(save_to_file)

    print()
    save_to_file = f"{saving_folder}/train_test_validation_prec_rec_curve.png" if save_figures else None
    _train_test_validation_metric_curve(df_train_probas, df_test_probas, df_val_probas,
                                       _precision_recall, secundary_color, 
                                       save_to_file=save_to_file)
    files_names.append(save_to_file)

    print()
    save_to_file = f"{saving_folder}/train_test_validation_cum_gains_curve.png" if save_figures else None
    _train_test_validation_metric_curve(df_train_probas, df_test_probas, df_val_probas,
                                       _cumulative_gain, secundary_color, 
                                       save_to_file=save_to_file)
    files_names.append(save_to_file)

    print()
    save_to_file = f"{saving_folder}/train_test_validation_lift_curve.png" if save_figures else None
    _train_test_validation_metric_curve(df_train_probas, df_test_probas, df_val_probas,
                                       _lift_curve, secundary_color, 
                                       save_to_file=save_to_file)
    files_names.append(save_to_file)


    if(n_percentiles):
        print()
        save_to_file = f"{saving_folder}/train_test_validation_ordenation_graphs.png" if save_figures else None
        save_rating_limits_to_file = f"{saving_folder}/train_test_validation_ordenation_rating_limits.txt" if save_figures else None
        _train_test_validation_ordenation_graphs(df_train_probas, df_test_probas, df_val_probas, 
                                                n_percentiles,
                                                stressed_color,secundary_color, 
                                                save_img_to_file=save_to_file, 
                                                save_rating_limits_to_file=save_rating_limits_to_file)
        files_names.append(save_to_file)
        files_names.append(save_rating_limits_to_file)
    if save_figures:
        return files_names

def train_test_validation_shap_graph(model, X_train, X_test, X_val,
                                     save_figure=False, saving_folder='.',
                                     return_figure=False):

    if isinstance(model,Pipeline):
        if hasattr(model[:-1],'transform'):
            X_train_transformed = model[:-1].transform(X_train)
            X_test_transformed = model[:-1].transform(X_test)
            X_val_transformed = model[:-1].transform(X_val)
        else:
            X_train_transformed = X_train
            X_test_transformed = X_test
            X_val_transformed = X_val
        model_aux = model[-1]
    else:
        X_train_transformed = X_train
        X_test_transformed = X_test
        X_val_transformed = X_val
        model_aux = model

    names_list = ['treino','teste','validação']
    Xs = [X_train_transformed, X_test_transformed, X_val_transformed]
    dict_exp_shap_values = dict()
    for name, X in zip(names_list,Xs):
        expected_value, shap_values = (
            calculate_shap_values(model_aux, X)
        )
        dict_exp_shap_values[name] = (X, expected_value, shap_values)

    # n_rows_plot = 1+len(cat_col)
    n_rows_plot = 1
    n_cols_plot = len(names_list)
    fig = plt.figure(figsize=(30,5*n_rows_plot))

    for i, name  in enumerate(names_list, start=1):
        ax = plt.subplot(n_rows_plot,n_cols_plot,i)
        X, expected_value, shap_values = dict_exp_shap_values[name]
        _beeswarm_from_shap_values(shap_values, X)
        ax.set_title(f'Valores SHAP ({name})')
        ax.set_xlabel('')

    plt.tight_layout()

    save_to_file = None
    filename = 'shap_results.png'
    if save_figure:
        save_to_file = f"{saving_folder}/{filename}"
        plt.savefig(save_to_file)

    plt.show()

    if return_figure:
        return {filename: fig}

    return save_to_file

def _train_test_validation_volumetry_performance(
    df_train_probas,
    df_test_probas,
    df_val_probas,
    save_to_file=None,
    return_fig=False
):

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("Inadimplência e Volume", "Desempenho",
                                        "Desempenho"),
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}],
                               [{"secondary_y": False}]])

    count_func = lambda x: len(x['target'])
    default_func = lambda x: x['target'].mean()
    roc_auc_func = lambda x: roc_auc_score(x['target'],x['probas'])
    ks_func = lambda x: ks_2samp(x['probas'][x['target'] == 0], x['probas'][x['target'] == 1]).statistic

    for df, group,line_type  in zip([df_train_probas,df_test_probas,df_val_probas],
                                    ['Treino','Teste','Validação'],
                                    ['solid','dash','dot']):
        df_aux = (
            df
            .groupby(['time'])
            .apply(lambda x: pd.Series({'Volume': count_func(x),
                                        'Inadimplência':default_func(x)}),
                    include_groups=False)
            .reset_index()
            .sort_values('time')
        )

        line_default = go.Scatter(x=df_aux['time'].dt.strftime('%m/%Y'),
                                  y=df_aux['Inadimplência'],
                                  name=f"Inadimplência ({group})",
                                  marker_color='blue',
                                  line={'dash':line_type},
                                  marker={'opacity':0})
        line_volume = go.Scatter(x=df_aux['time'].dt.strftime('%m/%Y'), 
                                 y=df_aux['Volume'],
                                 name=f"Volume ({group})",
                                 mode='lines',marker_color='red',
                                 line={'dash':line_type},
                                 marker={'opacity':0})
        fig.add_trace(line_default, row=1, col=1)
        fig.add_trace(line_volume, row=1, col=1, secondary_y=True)

        df_performance = (
            df
            .groupby(['time'])
            .apply(lambda x: pd.Series({'AUC': roc_auc_func(x),
                                        'KS': ks_func(x)}),
                   include_groups=False)
            .reset_index()
            .sort_values('time')
        )

        line_auc = go.Scatter(x=df_performance['time'].dt.strftime('%m/%Y'),
                          y=df_performance['AUC'], 
                          name=f"AUC ({group})",
                          marker_color='blue',
                          line={'dash':line_type},marker={'opacity':0})
        fig.add_trace(line_auc, row=2, col=1)

        line_ks = go.Scatter(x=df_performance['time'].dt.strftime('%m/%Y'),
                             y=df_performance['KS'],
                             name=f"KS ({group})",
                             marker_color='orange',
                             line={'dash':line_type},marker={'opacity':0})
        fig.add_trace(line_ks, row=3, col=1)

    fig.update_layout(
        title_text="Gráficos de performance",
        showlegend=False,
        yaxis=dict(range=[0,None],tickformat='.1%',title='Inadimplência'),
        yaxis2=dict(range=[0,None],title='Volume'),
        yaxis3=dict(tickformat='.1%',title='AUC'),
        yaxis4=dict(tickformat='.1%',title='KS'),
        width=800,
        height=600
    )
    fig.show()
    if save_to_file:
        fig.write_image(f"{save_to_file}")
    if return_fig:
        return fig

def _train_test_validation_volumetry_performance_by_group(
    df_train_probas,
    df_test_probas,
    df_val_probas,
    save_to_file=None,
    return_fig=False,
):

    count_func = lambda x: len(x['target'])
    default_func = lambda x: x['target'].mean()
    roc_auc_func = lambda x: (
        roc_auc_score(x['target'],x['probas']) 
            if x['target'].nunique() > 1 else 0.0
    )
    ks_func = lambda x: (
        ks_2samp(x['probas'][x['target'] == 0], x['probas'][x['target'] == 1]).statistic 
            if x['target'].nunique() > 1 else 0.0
    )

    fig = make_subplots(rows=4, cols=1,
                    subplot_titles=("Volume", "Inadimplência", "AUC",
                                    "KS"))

    for df, group,line_type  in zip([df_train_probas,df_test_probas,df_val_probas],
                                    ['Treino', 'Teste','Validação'],
                                    ['solid','dash','dot']):
        df_aux = (
            df
            .groupby(['time','group'])
            .apply(lambda x: pd.Series({'Volume': count_func(x),
                                        'Inadimplência':default_func(x),
                                        'AUC': roc_auc_func(x),
                                        'KS': ks_func(x)
                                        }),
                    include_groups=False)
            .reset_index()
            .sort_values('time')
        )

        for tn,marker_color in zip(df_aux['group'].unique(),['blue','red','green',
                                                               'orange','purple','black']):
            df_aux_tn = df_aux.query(f'group=="{tn}"')
            line_volume = go.Scatter(x=df_aux_tn['time'].dt.strftime('%m/%Y'), 
                                        y=df_aux_tn['Volume'],
                                        name=f"Volume ({tn},{group})",
                                        mode='lines',
                                        marker_color=marker_color,
                                        line={'dash':line_type},
                                        marker={'opacity':0})
            fig.add_trace(line_volume, row=1, col=1)
            line_default = go.Scatter(x=df_aux_tn['time'].dt.strftime('%m/%Y'),
                                        y=df_aux_tn['Inadimplência'],
                                        name=f"Inadimplência ({tn},{group})",
                                        marker_color=marker_color,
                                        line={'dash':line_type},
                                        marker={'opacity':0})
            fig.add_trace(line_default, row=2, col=1)

            line_auc = go.Scatter(x=df_aux_tn['time'].dt.strftime('%m/%Y'),
                                y=df_aux_tn['AUC'],
                                name=f"AUC ({tn},{group})",
                                marker_color=marker_color,
                                line={'dash':line_type},marker={'opacity':0})
            fig.add_trace(line_auc, row=3, col=1)
            line_ks = go.Scatter(x=df_aux_tn['time'].dt.strftime('%m/%Y'),
                                    y=df_aux_tn['KS'],
                                    name=f"KS ({tn},{group})",
                                    marker_color=marker_color,
                                    line={'dash':line_type},marker={'opacity':0})
            fig.add_trace(line_ks, row=4, col=1)

    fig.update_layout(
        title_text="Gráficos de performance",
        showlegend=False,
        yaxis=dict(range=[0,None],title='Volume'),
        yaxis2=dict(range=[0,None],tickformat='.1%',title='Inadimplência'),
        yaxis3=dict(tickformat='.1%',title='AUC'),
        yaxis4=dict(tickformat='.1%',title='KS'),
        width=800,
        height=300*4
    )
    fig.show()
    if save_to_file:
        fig.write_image(f"{save_to_file}")
    if return_fig:
        return fig

def _train_test_validation_ordenation_graphs_new(
    df_train_probas,
    df_test_probas,
    df_val_probas,
    save_to_file=None,
    return_fig=False
):
    
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Treino", "Teste", "Validação"),
                        specs=[[{"secondary_y": True}, {"secondary_y": True},
                                {"secondary_y": True}]])

    # Create bar and line charts
    max_default = 0
    max_vol = 0
    for i,df  in enumerate([df_train_probas,df_test_probas,df_val_probas]):
        df_aux = (
            df.groupby(['ratings'])
            .agg(
                **{'Inadimplência': ('target', 'mean'),
                'Volume': ('target','count')}
            )
            .reset_index()
        )
        text_default = df_aux['Inadimplência'].apply(lambda x: f"{x:.1%}")
        text_volume = (
            df_aux['Volume'].astype('string') +
            (df_aux['Volume']/df_aux['Volume'].sum()).apply(lambda x: f"({x:.1%})")
        )
        bar = go.Bar(x=df_aux['ratings'], 
                     y=df_aux['Inadimplência'],
                     name="Inadimplência",
                     yaxis=f'y{i*2+1}',
                     marker_color='blue',
                     text=text_default)
        line = go.Scatter(x=df_aux['ratings'],
                          y=df_aux['Volume'],
                          name="Volume", yaxis=f'y{i*2+2}',
                          mode='lines+text',marker_color='red',
                          marker={'angle':90},
                          text=text_volume,
                          textposition='top center',
                          textfont={'size':9,'color':'red'})
        fig.add_trace(bar, row=1, col=i+1)
        fig.add_trace(line, row=1, col=i+1, secondary_y=True)
        max_default = (
            df_aux['Inadimplência'].max() 
                if df_aux['Inadimplência'].max() > max_default 
                    else max_default
        )
        max_vol = df_aux['Volume'].max() if df_aux['Volume'].max()>max_vol else max_vol

    fig.update_layout(
        title_text="Gráficos de ordenação",
        showlegend=False,
        yaxis=dict(range=[0, max_default+0.01], tickformat='.1%', title='Inadimplência'),      # Set y-axis limits for bar charts
        yaxis2=dict(range=[0, max_vol+100], showticklabels=True, title='Volume'),     # Ensure secondary y-axes also have the same limits
        yaxis3=dict(range=[0, max_default+0.01], showticklabels=True, tickformat='.1%', title='Inadimplência'),
        yaxis4=dict(range=[0, max_vol+100], showticklabels=True, title='Volume'),
        yaxis5=dict(range=[0, max_default+0.01], showticklabels=True, tickformat='.1%', title='Inadimplência'),
        yaxis6=dict(range=[0, max_vol+100], title='Volume'),
        width=1500
    )

    fig.show()

    if save_to_file:
        fig.write_image(f"{save_to_file}")
    if return_fig:
        return fig
def train_test_validation_metrics_new(y_train, y_test, y_val,
                                      y_probas_train, y_probas_test, y_probas_val, 
                                      cohort_train, cohort_test, cohort_val,
                                      rating_train=None, rating_test=None,
                                      rating_val=None,
                                      save_figures=False,
                                      saving_folder='.',
                                      return_figures=False,
                                      group_train=None,
                                      group_test=None,
                                      group_val=None,):

    df_train_probas = pd.DataFrame()
    df_test_probas = pd.DataFrame()
    df_val_probas = pd.DataFrame()

    df_train_probas['target'] = y_train.reset_index(drop=True)
    df_test_probas['target'] = y_test.reset_index(drop=True)
    df_val_probas['target'] = y_val.reset_index(drop=True)

    df_train_probas['probas'] = y_probas_train
    df_test_probas['probas'] = y_probas_test
    df_val_probas['probas'] = y_probas_val

    df_train_probas['time'] = cohort_train.reset_index(drop=True)
    df_test_probas['time'] = cohort_test.reset_index(drop=True)
    df_val_probas['time'] = cohort_val.reset_index(drop=True)

    if (
        group_train is not None and
        group_test is not None and
        group_val is not None
    ):
        df_train_probas['group'] = group_train.reset_index(drop=True)
        df_test_probas['group'] = group_test.reset_index(drop=True)
        df_val_probas['group'] = group_val.reset_index(drop=True)
    # stressed_color = '#006e9cff'
    secundary_color = 'gray'

    files_names = []

    figs = {}

    print()
    auc_train = _calculate_auc(df_train_probas)
    auc_test = _calculate_auc(df_test_probas)
    auc_val = _calculate_auc(df_val_probas)
    print(f'ROC AUC (treino): {auc_train}')
    print(f'ROC AUC (teste): {auc_test}')
    print(f'ROC AUC (validação): {auc_val}')

    print()
    filename = 'train_test_validation_volumetry_auc_over_time.html'
    save_to_file = f"{saving_folder}/{filename}" if save_figures else None
    figs[filename] = (
        _train_test_validation_volumetry_performance(df_train_probas,
                                                     df_test_probas,
                                                     df_val_probas,
                                                     save_to_file=save_to_file,
                                                     return_fig=return_figures)
    )
    files_names.append(save_to_file)

    print()
    filename = 'train_test_validation_metric_curve.png'
    save_to_file = f"{saving_folder}/{filename}" if save_figures else None
    figs[filename] = (
        _train_test_validation_metric_curve(df_train_probas, df_test_probas, 
                                            df_val_probas,
                                            _roc, secundary_color, 
                                            save_to_file=save_to_file,
                                            return_fig=return_figures)
    )
    files_names.append(save_to_file)

    print()
    filename = 'train_test_validation_ks_curve.png'
    save_to_file = f"{saving_folder}/{filename}" if save_figures else None
    figs[filename] = (
        _train_test_validation_metric_curve(df_train_probas, df_test_probas, 
                                            df_val_probas,
                                            _ks_statistic, secundary_color, 
                                            save_to_file=save_to_file,
                                            return_fig=return_figures)
    )
    files_names.append(save_to_file)

    print()
    filename = 'train_test_validation_prec_rec_curve.png'
    save_to_file = f"{saving_folder}/{filename}" if save_figures else None
    figs[filename] = (
        _train_test_validation_metric_curve(df_train_probas, df_test_probas, 
                                            df_val_probas,
                                            _precision_recall, secundary_color, 
                                            save_to_file=save_to_file,
                                            return_fig=return_figures)
    )
    files_names.append(save_to_file)

    print()
    filename = 'train_test_validation_cum_gains_curve.png'
    save_to_file = f"{saving_folder}/{filename}" if save_figures else None
    figs[filename] = (
        _train_test_validation_metric_curve(df_train_probas, df_test_probas,
                                            df_val_probas,
                                            _cumulative_gain, secundary_color,
                                            save_to_file=save_to_file,
                                            return_fig=return_figures)
    )
    files_names.append(save_to_file)

    print()
    filename = 'train_test_validation_lift_curve.png'
    save_to_file = f"{saving_folder}/{filename}" if save_figures else None
    figs[filename] = (
        _train_test_validation_metric_curve(df_train_probas, 
                                            df_test_probas,
                                            df_val_probas,
                                            _lift_curve, secundary_color, 
                                            save_to_file=save_to_file,
                                            return_fig=return_figures)
    )
    files_names.append(save_to_file)

    if ((rating_train is not None) and 
        (rating_test is not None) and 
        (rating_val is not None)):
        df_train_probas['ratings'] = rating_train.reset_index(drop=True)
        df_test_probas['ratings'] = rating_test.reset_index(drop=True)
        df_val_probas['ratings'] = rating_val.reset_index(drop=True)
        print()
        filename = 'train_test_validation_ordenation_graphs.html'
        save_to_file = f"{saving_folder}/{filename}" if save_figures else None
        figs[filename] = (
            _train_test_validation_ordenation_graphs_new(df_train_probas,
                                                         df_test_probas,
                                                         df_val_probas,
                                                         save_to_file=save_to_file,
                                                         return_fig=return_figures)
        )
        files_names.append(save_to_file)
    if (
        group_train is not None and
        group_test is not None and
        group_val is not None
    ):
        print()
        filename = 'train_test_validation_volumetry_auc_over_time_by_group.html'
        save_to_file = f"{saving_folder}/{filename}" if save_figures else None
        figs[filename] = (
            _train_test_validation_volumetry_performance_by_group(df_train_probas,
                                                                  df_test_probas,
                                                                  df_val_probas,
                                                                  save_to_file=save_to_file,
                                                                  return_fig=return_figures)
        )
        files_names.append(save_to_file)

    if return_figures:
        return figs
    if save_figures:
        return files_names
