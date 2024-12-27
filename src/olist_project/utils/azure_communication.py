import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


####################################################################################
################################ azure blob storage ################################
####################################################################################

def az_read_parquet(
        file_name: str,
        container_name: str = 'data/transient/ipiranga_empresas/smart_credit_ie' ,
        resource: str = 'stippdatalakelab',
        ) -> pd.DataFrame:

    storage_options = {
        'account_name': resource,
        'anon': False
    }
    full_path = f"abfs://{container_name}/{file_name}.parquet"
    df = pd.read_parquet(full_path, storage_options=storage_options)

    return df

def get_client(ACC_URL):

    # Obtém a credencial padrão do Azure
    credential = DefaultAzureCredential()

    # Cria um cliente de serviço de blob com a URL da conta e a credencial
    blob_service_client = BlobServiceClient(ACC_URL, credential=credential)

    return blob_service_client

def get_blob_list(
        container_name: str = 'data' ,
        resource: str = 'transient/ipiranga_empresas/smart_credit_ie',
        ) -> list:

    blob_service_client = get_client("https://stippdatalakelab.blob.core.windows.net/")
    blob_list = blob_service_client.get_container_client(container_name).list_blobs(name_starts_with=resource)
    files = {blob.name: blob.last_modified for blob in blob_list if blob.name.endswith('.parquet')}

    return files
