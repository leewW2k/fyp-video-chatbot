from typing import List, Any, Mapping

import numpy as np
from bson import ObjectId
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate
from openai import AsyncAzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity

from database_service import AzureDatabaseService
from utils import convert_seconds_to_mm_ss, process_file


class OpenAIService:
    """
    OpenAIService is a wrapper class of AsyncAzureOpenAI used for generating responses from Azure OpenAI LLM.
    It provides the ability to add system message, grounding text, and a generic prompt template.

    Args:
        azure_endpoint (str): Azure OpenAI Endpoint. Example: https://<YOUR_RESOURCE_NAME>.openai.azure.com/. Required.
        api_key (str): Azure OpenAI API Key. Required.
        deployment_name (str): Azure OpenAI Deployment Name. Example: gpt-4o-mini. Required.
        database_service (AzureDatabaseService): Instance of AzureDatabaseService class. Required.
        prompt_template_fp (str): Filepath to prompt template to be used in chatbot prompt. Default: "prompt_template.txt".
        temperature (float): Chatbot Temperature. Default: 0.
        embedding_model (str): Embedding Model. Default: "all-MiniLM-L6-v2".
    """
    def __init__(
            self,
            azure_endpoint: str,
            api_key: str,
            deployment_name: str,
            api_version : str,
            database_service: AzureDatabaseService,
            prompt_template_fp: str="prompt_template.txt",
            temperature: float=0,
            embedding_model: str="all-MiniLM-L6-v2"
    ):
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.database_service = database_service
        self.client = self.initiate_client()
        try:
            self.prompt_template = process_file(fp=prompt_template_fp, mode='r')
        except Exception as e:
            print(e)
            self.prompt_template = ""
        self.embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model)
        self.chat_model = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            azure_deployment=deployment_name,
            temperature=temperature
        )

    def initiate_client(self):
        """
        Initialises a AsyncAzureOpenAI instance to interact with Azure OpenAI services.
        Uses provided Azure endpoint, API key, and API version.
        If an error occurs during initialization, the Azure endpoint and the exception are printed for debugging.

        Returns:
            AsyncAzureOpenAI: An initialised instance of the AsyncAzureOpenAI class.

        Raises:
            Exception: For any errors that occur during initialization.
        """
        try:
            return AsyncAzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        except Exception as ex:
            print(ex)

    async def generate_video_prompt_response(self, user_prompt: str, video_id: str):
        """
        Generate response based on user prompt and selected video.

        Args:
            user_prompt (str): User prompt to query chatbot. Required.
            video_id (str): Video ID of the video selected. Required.
        """
        try:
            prompt = PromptTemplate(
                template=self.prompt_template,
                input_variables=["context", "input"]
            )

            # Perform a retrieval with metadata
            args = {"video_reference_id": ObjectId(video_id)}
            retrieval_results = self.database_service.retrieve_results_embedding(
                kwargs={"text": 1, "embedding": 1, "metadata": 1, "frames": 1},
                args=args
            )

            # Calculate cosine similarity
            similarities = []
            query_embedding = np.array(self.embedding_function.embed_query(user_prompt))
            for doc in retrieval_results:
                doc_embedding = np.array(doc["embedding"])
                similarity_score = cosine_similarity(query_embedding.reshape(1, -1), doc_embedding)[0][0]
                similarities.append((similarity_score, doc))

            # Sort and display top results
            similarities.sort(reverse=True, key=lambda x: x[0])

            documents = similarities[:5]

            context = self.map_mongo_documents_to_prompt_context(documents)

            # Extract context and metadata (timestamps) from the retrieved documents
            documents = [Document(page_content=context)]

            combine_docs_chain = create_stuff_documents_chain(self.chat_model, prompt)

            # Generate the full prompt with context and input
            final_prompt = prompt.format(context=context, input=user_prompt)

            # Save the prompt to a file
            process_file(fp="generated_prompt.txt", mode="w", content=final_prompt)

            # Return the response
            return combine_docs_chain.invoke({
                "context": documents,
                "input": user_prompt
            })
        except Exception as ex:
            print(ex)
            return ex

    def map_mongo_documents_to_prompt_context(self, documents: List[tuple[Any, Mapping[str, Any]]]) -> str:
        """
        Generate context from mongo documents retrieved.

        Args:
            documents (List[tuple[Any, Mapping[str, Any]]]): Documents from mongoDB.

        Returns:
            Context to be passed to prompt
        """
        context = ""
        for _, doc in documents:
            frames = doc.get("frames", [])

            frame_details = "\n".join([
                f"  - Frame at {convert_seconds_to_mm_ss(frame.get('timestamp', 0))}: {frame.get('tag', {}).get('caption', {}).get('text', '')}\n"
                f"    Dense Captions: {', '.join([caption.get('text', '') for caption in frame.get('tag', {}).get('dense_captions', [])])}\n"
                f"    Lines: {', '.join([read.get('line', '') for read in frame.get('tag', {}).get('read', [])])}\n"
                f"    Tags: {', '.join([tag.get('name', '') for tag in frame.get('tag', {}).get('tags', [])])}\n"
                f"    Objects: {', '.join([obj.get('name', '') for obj in frame.get('tag', {}).get('objects', [])])}\n"
                for frame in frames
            ])

            context += f"Transcript: {doc['text']}\n"
            if frames:
                context += f"Associated Frames:\n{frame_details}\n"
        return context
