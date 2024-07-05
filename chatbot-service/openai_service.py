# Add OpenAI library
import os
from openai import AzureOpenAI
import logging


class OpenAIService:
    def __init__(self):
        self.azure_endpoint = os.environ.get("YOUR_ENDPOINT_NAME")
        self.api_key = os.environ.get("YOUR_API_KEY")
        self.deployment_name = os.environ.get("YOUR_DEPLOYMENT_NAME")
        self.api_version = os.environ.get("YOUR_API_VERSION")
        self.client = self.initiate_client()
        self.logger = logging.getLogger()

    def initiate_client(self):
        try:
            return AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
        except Exception as ex:
            self.logger.warning(ex)

    def create_response(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        generated_text = response.choices[0].message.content

        # Print the response
        print("Response: " + generated_text + "\n")

    def create_prompt_response(self):
        while True:
            # Get input text
            input_text = input("Enter the prompt (or type 'quit' to exit): ")
            if input_text.lower() == "quit":
                break
            if len(input_text) == 0:
                print("Please enter a prompt.")
                continue

            print("\nSending request for summary to Azure OpenAI endpoint...\n\n")
            self.create_response(input_text)
