"""
This is a boilerplate pipeline 'ingesting'
generated using Kedro 0.19.10
"""

from kagglehub.pandas_datasets import load_pandas_dataset

def ingest(dasets_schema):
    """
    """
    dfs = []
    for dataset_name,schema in dasets_schema.items():
        df = load_pandas_dataset("olistbr/brazilian-ecommerce/versions/2",
                                path=f'olist_{dataset_name}_dataset.csv',
                                sql_query=f'SELECT * FROM olist_{dataset_name}_dataset')
        df = df.astype(schema)
        dfs.append(df)
    return dfs
