from abc import ABC

import requests
from langchain_core.document_loaders.base import BaseBlobParser


class AzureOpenAIWhisperParser(BaseBlobParser, ABC):
    def __init__(self, api_base, api_key):
        self.api_base = api_base
        self.api_key = api_key

    def transcribe(self, audio_path):
        url = f"{self.api_base}/openai/deployments/whisper/transcriptions?api-version=2023-01-01-preview"
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/octet-stream"
        }
        with open(audio_path, "rb") as audio_file:
            response = requests.post(url, headers=headers, data=audio_file)
        if response.status_code == 200:
            return response.json()["text"]
        else:
            raise Exception(f"Error in transcription: {response.text}")

    def parse(self, audio_path):
        text = self.transcribe(audio_path)
        return [{"page_content": text}]