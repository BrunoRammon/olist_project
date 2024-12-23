"""
This is a boilerplate pipeline 'ingesting'
generated using Kedro 0.19.10
"""

import kagglehub
import pandas as pd
import os

def ingest():
    """
    """
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    source_dir = path
    dfs = []
    for filename in sorted(os.listdir(source_dir)):
        source_file = os.path.join(source_dir, filename)
        dfs.append(pd.read_csv(source_file))
    return dfs
