from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        feature_disc = pd.cut(df[feature], bins=bins_feature, labels=None, include_lowest=True)
        if df[feature].isna().sum() > 0:
            feature_disc = feature_disc.astype('category')
            feature_disc = feature_disc.cat.add_categories('Nulo')
            feature_disc = feature_disc.fillna('Nulo')
    
    elif isinstance(df[feature].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(df[feature]):
        feature_disc = df[feature].astype(str)

    return feature_disc

def _calculate_volumes_features_groups(
    df: pd.DataFrame,
    features: List[str],
    period_column: str,
    mask_reference: pd.Series,
    num_bins: int = 5,
) -> pd.DataFrame:


    dfs_features = []

    period_data = df[mask_reference].copy()
    df = (
        df
        # [~mask_reference]
        .assign(
            flag_reference = mask_reference.astype(int),
        )
        .copy()
    )
    total_volume_by_safra = (
        df.groupby(['flag_reference',period_column]).size()
        .reset_index(name=f'VOLUME_{period_column}')
    )
    for feature in features:
        df['GRUPO'] = _calculate_bins_per_feature(df, feature, period_data, num_bins=num_bins)

        metrics_df = (
            df
            .groupby(['flag_reference', period_column, 'GRUPO'], observed=True)
            .agg(
                VOLUME_TOTAL=(period_column, 'size'),
            )
            .reset_index()
        )

        metrics_df = metrics_df.merge(total_volume_by_safra, on=['flag_reference',period_column])
        metrics_df['VOLUME_PERCENTUAL'] = (metrics_df['VOLUME_TOTAL'] / metrics_df[f'VOLUME_{period_column}'])
        metrics_df['VARIAVEL'] = feature
        result_df = (
            metrics_df
            [['flag_reference', period_column, 'VARIAVEL', 'GRUPO', 'VOLUME_TOTAL', 'VOLUME_PERCENTUAL']]
            .assign(
                GRUPO = lambda df: df.GRUPO.astype(str)
            )
        )
        dfs_features.append(result_df)

    df_group_feature_volumes = (
        pd.concat(dfs_features, ignore_index=True)
    )

    return df_group_feature_volumes

def _calculate_psi(expected, actual):
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
) -> pd.DataFrame:

    period_data = df[df.flag_reference == 1]
    n_samples_reference = period_data[period_data['VARIAVEL'] == features[0]].VOLUME_TOTAL.sum()

    psi_results = []

    for feature in features:
        expected_counts = (
            period_data[period_data['VARIAVEL'] == feature]
            .groupby('GRUPO').VOLUME_TOTAL.sum()
        ) / n_samples_reference
        for period in df[df.flag_reference != 1][period_column].unique():
            actual_counts = (
                df[(df[period_column] == period) & 
                   (df['VARIAVEL'] == feature) &
                   (df.flag_reference != 1)]
                   [['GRUPO', 'VOLUME_PERCENTUAL']]
                .set_index('GRUPO')['VOLUME_PERCENTUAL']
            )

            combined_counts = pd.concat([expected_counts, actual_counts], axis=1).fillna(0)
            combined_counts.columns = ['expected', 'actual']
            psi_value = _calculate_psi(combined_counts['expected'], combined_counts['actual'])
            psi_results.append({
                'SAFRA': period,
                'VARIAVEL': feature,
                'PSI': psi_value
            })

    psi_df = pd.DataFrame(psi_results)

    return psi_df


def calculate_psi(data, month_column,
                  reference_mask,
                  numerical_features,
                  categorical_features,
                  bins=10,
                  epsilon=1e-10):
    
    # Step 1: Filter the reference period data
    reference_data = data[reference_mask].copy()
    data = data[~reference_mask].copy()
    # Create an empty list to store PSI results
    psi_results = []

    # Calculate the bins based on the reference data for numerical features
    bin_dict = {}
    for feature in numerical_features:
        ref_dist = _define_numeric_bins_limits(reference_data, feature, bins)
        bin_dict[feature] = ref_dist

    # Calculate the unique categories based on the entire dataframe for categorical features
    cat_dict = {}
    for feature in categorical_features:
        unique_cats = data[feature].unique()
        cat_dict[feature] = unique_cats

    # Step 2: Calculate PSI for each feature in each month
    for month in sorted(data[month_column].unique().tolist()):
        # Filter data for the current month
        month_data = data[data[month_column] == month]

        for feature in data.columns:
            # Identify the feature type (categorical or numeric)
            if feature in categorical_features:
                # Calculate percentage distribution in reference and current month for categorical feature
                ref_dist = reference_data[feature].value_counts(normalize=True, sort=False) + epsilon
                comp_dist = month_data[feature].value_counts(normalize=True, sort=False) + epsilon

                # Use the same set of categories across all months
                ref_dist = ref_dist.reindex(cat_dict[feature], fill_value=epsilon)
                comp_dist = comp_dist.reindex(cat_dict[feature], fill_value=epsilon)
            elif feature in numerical_features:
                # Calculate percentage distribution in reference and current month for numeric feature
                ref_dist = pd.cut(reference_data[feature], bins=bin_dict[feature], include_lowest=True)
                ref_dist = ref_dist.astype('category')
                ref_dist = ref_dist.cat.add_categories('Nulo')
                ref_dist = ref_dist.fillna('Nulo')
                ref_dist = ref_dist.value_counts(normalize=True, sort=False) + epsilon

                comp_dist = pd.cut(month_data[feature], bins=bin_dict[feature], include_lowest=True)
                comp_dist = comp_dist.astype('category')
                comp_dist = comp_dist.cat.add_categories('Nulo')
                comp_dist = comp_dist.fillna('Nulo')
                comp_dist = comp_dist.value_counts(normalize=True, sort=False) + epsilon
            else:
                continue  # Skip if the feature is neither numerical nor categorical

            # Merge reference and current month distributions
            dist_df = pd.concat([ref_dist, comp_dist], axis=1, keys=['Reference', 'Comparison']).fillna(0)

            # Calculate PSI for the feature in the current month
            psi = sum((dist_df['Reference'] - dist_df['Comparison']) * np.log(dist_df['Reference'] / dist_df['Comparison']))

            # Append the PSI result to the list
            psi_results.append({'Feature': feature, 'Safra': month, 'PSI': psi})

    # Step 3: Convert the PSI results to a dataframe
    psi_df = pd.DataFrame(psi_results)

    return psi_df

def calculate_psi_2(data, month_column,
                  reference_mask,
                  numerical_features,
                  categorical_features,
                  bins=10,
                  epsilon=1e-10):
    
    # Step 1: Filter the reference period data
    data_all = data.copy()
    reference_data = data[reference_mask].copy()
    data = data[~reference_mask].copy()
    # Create an empty list to store PSI results
    psi_results = []

    # Calculate the bins based on the reference data for numerical features
    bin_dict = {}
    for feature in numerical_features:
        ref_dist = _define_numeric_bins_limits(data_all, feature, bins)
        bin_dict[feature] = ref_dist

    # Calculate the unique categories based on the entire dataframe for categorical features
    cat_dict = {}
    for feature in categorical_features:
        unique_cats = data[feature].unique()
        cat_dict[feature] = unique_cats

    # Step 2: Calculate PSI for each feature in each month
    for month in sorted(data[month_column].unique().tolist()):
        # Filter data for the current month
        month_data = data[data[month_column] == month]

        for feature in data.columns:
            # Identify the feature type (categorical or numeric)
            if feature in categorical_features:
                # Calculate percentage distribution in reference and current month for categorical feature
                ref_dist = reference_data[feature].value_counts(normalize=True, sort=False) + epsilon
                comp_dist = month_data[feature].value_counts(normalize=True, sort=False) + epsilon

                # Use the same set of categories across all months
                ref_dist = ref_dist.reindex(cat_dict[feature], fill_value=epsilon)
                comp_dist = comp_dist.reindex(cat_dict[feature], fill_value=epsilon)
            elif feature in numerical_features:
                # Calculate percentage distribution in reference and current month for numeric feature
                ref_dist = pd.cut(reference_data[feature], bins=bin_dict[feature], include_lowest=True)
                ref_dist = ref_dist.astype('category')
                ref_dist = ref_dist.cat.add_categories('Nulo')
                ref_dist = ref_dist.fillna('Nulo')
                ref_dist = ref_dist.value_counts(normalize=True, sort=False) + epsilon

                comp_dist = pd.cut(month_data[feature], bins=bin_dict[feature], include_lowest=True)
                comp_dist = comp_dist.astype('category')
                comp_dist = comp_dist.cat.add_categories('Nulo')
                comp_dist = comp_dist.fillna('Nulo')
                comp_dist = comp_dist.value_counts(normalize=True, sort=False) + epsilon
            else:
                continue  # Skip if the feature is neither numerical nor categorical

            # Merge reference and current month distributions
            dist_df = pd.concat([ref_dist, comp_dist], axis=1, keys=['Reference', 'Comparison']).fillna(0)

            # Calculate PSI for the feature in the current month
            psi = sum((dist_df['Reference'] - dist_df['Comparison']) * np.log(dist_df['Reference'] / dist_df['Comparison']))

            # Append the PSI result to the list
            psi_results.append({'Feature': feature, 'Safra': month, 'PSI': psi})

    # Step 3: Convert the PSI results to a dataframe
    psi_df = pd.DataFrame(psi_results)

    return psi_df

def feature_psi_plots(df_psi, feat_name_col, safra_col, psi_col='PSI'):
    df_psi[safra_col] = df_psi[safra_col].astype(str)
    for var in df_psi[feat_name_col].unique().tolist():
        
        df_plot = df_psi[df_psi[feat_name_col] == var]
        
        plt.figure(figsize=(10, 6))  # Define o tamanho da figura
        plt.plot(df_plot[safra_col], df_plot[psi_col], marker='o', linestyle='-')  # Cria o gráfico de linha
        plt.title('PSI ao longo das Safras para ' + var)  # Define o título do gráfico
        plt.xlabel('Safra')  # Define o rótulo do eixo x
        plt.ylabel('PSI')  # Define o rótulo do eixo y
        plt.grid(True)  # Adiciona grades ao gráfico
        plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo x para melhorar a legibilidade
        
        y_lim = df_plot['PSI'].max() * 1.1
        y_lim = y_lim if y_lim > 0.25 else 0.25
        # Ajusta os limites do eixo y
        plt.ylim(0, y_lim)
        
        # Adiciona linhas tracejadas em y = 0.1 e y = 0.2
        plt.axhline(y=0.1, color='yellow', linestyle='--', label='Desvio não significativo')
        plt.axhline(y=0.2, color='red', linestyle='--', label='Desvio significativo')
        
        # plt.legend()  # Adiciona a legenda ao gráfico
        plt.tight_layout()  # Ajusta o layout para evitar que os rótulos se sobreponham
        plt.show()  # Mostra o gráfico