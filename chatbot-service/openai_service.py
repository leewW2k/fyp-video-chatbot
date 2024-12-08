# Add OpenAI library
import json
import os

import chromadb
import numpy as np
import pymongo
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from openai import AsyncAzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from utils import convert_seconds_to_mm_ss

load_dotenv()


class OpenAIService:
    def __init__(self):
        self.azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.deployment_name = os.environ.get("YOUR_DEPLOYMENT_NAME")
        self.api_version = os.environ.get("OPENAI_API_VERSION")
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
        self.embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        mongo_client = pymongo.MongoClient(os.environ.get("MONGODB_CONNECTION_STRING"))
        db = mongo_client["video_transcripts"]
        self.collection = db["transcripts"]
        self.videos_collection = db["videos"]


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
                template="""Given the context about a video. Answer the user queries with information from the video. The start time is given in the brackets ('[' and ']').
                Please output the 'start' time (in mm:ss format) from the context whenever you provide an answer.
                
                Context: {context}
                Human: {input}
                
                AI:""",
                input_variables=["context", "input"]
            )
            llm = AzureChatOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
                azure_deployment=self.deployment_name,
            )

            query_embedding = np.array(self.embedding_function.embed_query(user_prompt))

            # Perform a retrieval with metadata
            retrieval_results = self.collection.find({}, {"text": 1, "embedding": 1, "metadata": 1, "frames": 1})

            # Calculate cosine similarity
            similarities = []
            for doc in retrieval_results:
                doc_embedding = np.array(doc["embedding"])
                similarity_score = cosine_similarity(query_embedding.reshape(1, -1), doc_embedding)[0][0]
                similarities.append((similarity_score, doc))

            # Sort and display top results
            similarities.sort(reverse=True, key=lambda x: x[0])

            documents = similarities[:5]

            context = ""
            for _, doc in documents:
                start_time = convert_seconds_to_mm_ss(doc["metadata"].get("start", 0))
                frames = doc.get("frames", [])

                frame_details = "\n".join([
                    f"  - Frame at {convert_seconds_to_mm_ss(frame['timestamp'])}: {frame['tag']['caption']['text']}\n"
                    f"    Dense Captions: {', '.join([caption['text'] for caption in frame['tag']['dense_captions']])}"
                    for frame in frames
                ])

                context += f"[{start_time}] {doc['text']}\n"
                if frames:
                    context += f"Associated Frames:\n{frame_details}\n"

            # Extract context and metadata (timestamps) from the retrieved documents
            documents = [Document(page_content=context)]

            combine_docs_chain = create_stuff_documents_chain(llm, prompt)

            # Return the response
            return combine_docs_chain.invoke({
                "context": documents,
                "input": user_prompt
            })
        except Exception as ex:
            print(ex)
            return ex
