# Add OpenAI library
import os
from openai import AsyncAzureOpenAI
import logging


class OpenAIService:
    def __init__(self):
        self.azure_endpoint = os.environ.get("YOUR_ENDPOINT_NAME")
        self.api_key = os.environ.get("YOUR_API_KEY")
        self.deployment_name = os.environ.get("YOUR_DEPLOYMENT_NAME")
        self.api_version = os.environ.get("YOUR_API_VERSION")
        self.client = self.initiate_client()
        self.logger = logging.getLogger()
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        self.system_message = ""
        self.grounding_text = ""
        try:
            self.system_message = open(file="system.txt", encoding="utf8").read().strip()
        except Exception as ex:
            self.logger.warning(ex)
        try:
            self.grounding_text = open(file="grounding.txt", encoding="utf8").read().strip()
        except Exception as ex:
            self.logger.warning(ex)
        self.messages_array = [{"role": "system", "content": self.system_message}]

    def initiate_client(self):
        try:
            return AsyncAzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
        except Exception as ex:
            self.logger.warning(ex)

    async def create_response(self, prompt: str):
        response = await self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]
        )
        generated_text = response.choices[0].message.content

        # Print the response
        print("Response: " + generated_text + "\n")

    async def create_response_history(self, prompt: str):
        self.messages_array.append({"role": "user", "content": self.grounding_text + prompt})

        response = await self.client.chat.completions.create(
            model=self.deployment_name,
            messages=self.messages_array
        )
        generated_text = response.choices[0].message.content
        self.messages_array.append({"role": "assistant", "content": generated_text})

        # Print the response
        print("Response: " + generated_text + "\n")

    async def create_prompt_response(self, history=True):
        while True:
            # Get input text
            input_text = input("Enter the prompt (or type 'quit' to exit): ")
            if input_text.lower() == "quit":
                break
            if len(input_text) == 0:
                print("Please enter a prompt.")
                continue

            print("\nSending request for summary to Azure OpenAI endpoint...\n\n")
            self.logger.info("History Enabled: " + str(history))
            if history:
                self.messages_array = [{"role": "system", "content": self.system_message}]
                await self.create_response_history(input_text)
            else:
                await self.create_response(input_text)
