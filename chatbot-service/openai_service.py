# Add OpenAI library
import json
import os

import chromadb
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import AzureChatOpenAI

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from openai import AsyncAzureOpenAI

load_dotenv()


class OpenAIService:
    def __init__(self):
        self.azure_endpoint = os.environ.get("YOUR_ENDPOINT_NAME")
        self.api_key = os.environ.get("YOUR_API_KEY")
        self.deployment_name = os.environ.get("YOUR_DEPLOYMENT_NAME")
        self.api_version = os.environ.get("YOUR_API_VERSION")
        self.client = self.initiate_client()
        self.system_message = ""
        self.grounding_text = ""
        try:
            self.system_message = open(file="system.txt", encoding="utf8").read().strip()
        except Exception as ex:
            print(ex)
        try:
            self.grounding_text = open(file="grounding.txt", encoding="utf8").read().strip()
        except Exception as ex:
            print(ex)
        self.messages_array = [{"role": "system", "content": self.system_message}]
        self.persistent_client = chromadb.PersistentClient(path="./vector_store_003")
        self.embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vectordb = Chroma(
            embedding_function=self.embedding_function,
            persist_directory="vector_store_003",
            collection_name="test_video"
        )


    def initiate_client(self):
        try:
            return AsyncAzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
        except Exception as ex:
            print(self.azure_endpoint)
            print(ex)

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
        return generated_text

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
            print("History Enabled: " + str(history))
            if history:
                self.messages_array = [{"role": "system", "content": self.system_message}]
                await self.create_response_history(input_text)
            else:
                await self.create_response(input_text)

    async def generate_video_prompt_response(self, user_prompt: str):
        try:
            prompt = PromptTemplate(
                template="""Given the context about a video. Answer the user queries with information from the video.
                
                Context: {context}
                Timestamps: {timestamps}
                Human: {input}
                
                AI:""",
                input_variables=["context", "timestamps", "input"]
            )
            llm = AzureChatOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
                deployment_name="test-123",
            )

            # Perform a retrieval with metadata
            retrieval_results = self.vectordb.as_retriever().invoke(user_prompt)

            print(retrieval_results)

            # Extract context and metadata (timestamps) from the retrieved documents
            context = " ".join([f"[{doc.metadata.get('start')} - {doc.metadata.get('end')}]" + doc.page_content for doc in retrieval_results])
            timestamps = " ".join([f"[{doc.metadata.get('start')} - {doc.metadata.get('end')}]" for doc in retrieval_results])

            combine_docs_chain = create_stuff_documents_chain(llm, prompt)

            # Create RAG chain with context and timestamps
            rag_chain = create_retrieval_chain(
                retriever=self.vectordb.as_retriever(),
                combine_docs_chain=combine_docs_chain
            )

            # Return the response
            return rag_chain.invoke({
                "context": context,
                "timestamps": timestamps,
                "input": user_prompt
            })
        except Exception as ex:
            print(ex)
            return ex
