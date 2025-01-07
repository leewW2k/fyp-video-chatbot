import json

from azure.storage.blob import BlobServiceClient
from numpy.f2py.auxfuncs import throw_error


class AzureBlobService:
    """
    Class that interacts with Azure Blob.
    """
    def __init__(
            self,
            azure_storage_connection_string: str,
            azure_container_name: str,
            decode: str = "utf-8"
    ):
        self.azure_container_name = azure_container_name
        self.decode = decode
        self.blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)
        self.container_client = None
        self.blob_client = None

    def fetch_transcript_from_blob(self, blob_name):
        container_client = self.blob_service_client.get_container_client(self.azure_container_name)

        blob_client = container_client.get_blob_client(blob_name)
        if blob_client.exists():
            blob_content = blob_client.download_blob().readall()
            transcript_json = blob_content.decode(self.decode)
            transcript_list = json.loads(transcript_json)
            return transcript_list
        else:
            print(f"{blob_name} does not exist in Azure Blob Storage.")
            return None

    def create_container(self, container_name: str = None):
        if container_name is None:
            container_name = self.azure_container_name
        container_client = self.blob_service_client.get_container_client(container_name)
        try:
            container_client.create_container()
        except Exception as e:
            print(e)
            pass

    def get_container_client(self, container_name: str):
        self.container_client = self.blob_service_client.get_container_client(container_name)

    def get_blob_client(self, blob_name: str):
        if self.container_client is None:
            throw_error("No container client initialised")
        self.blob_client = self.container_client.get_blob_client(blob_name)

    def upload(self, json_input):
        if not self.blob_client.exists():
            self.blob_client.upload_blob(json_input, overwrite=True)