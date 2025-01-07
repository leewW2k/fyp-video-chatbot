from typing import Dict, Any

import pymongo
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings

from utils import convert_seconds_to_mm_ss

load_dotenv()


class AzureDatabaseService:
    """
    AzureDatabaseService is a service that interacts with Azure CosmosDB Vector store.

    Args:
        mongo_connection_string (str): MongoDB Connection String. Example: mongodb+srv://<username>:<password>@<resource>.mongocluster.cosmos.azure.com/<...>. Required.
        database_name (str): MongoDB Database Name. Required.
        video_collection_name (str): Azure OpenAI Deployment Name. Default: "videos".
        embedding_collection_name (str): Azure OpenAI API Version. Default: "transcripts".
        embedding_model (str): Embedding Model. Default: "all-MiniLM-L6-v2".
    """

    def __init__(
            self,
            mongo_connection_string: str,
            database_name: str,
            video_collection_name: str = "videos",
            embedding_collection_name: str = "transcripts",
            embedding_model: str = "all-MiniLM-L6-v2"
    ):
        mongo_client = pymongo.MongoClient(mongo_connection_string)
        db = mongo_client[database_name]
        self.collection = db[embedding_collection_name]
        self.videos_collection = db[video_collection_name]
        self.embedding_model = embedding_model

    def retrieve_results_embedding(self, kwargs: Dict[str, Any] = None, args: Dict[str, Any] = None):
        """
        Retrieve results from mongoDB embedding collection given keyword arguments and filter.

        Args:
            kwargs (Dict[str, Any]): Keyword Arguments for fields to retrieve from mongoDB. Default: None.
            args (Dict[str, Any]): Arguments to filter from mongoDB. Default: None.

        Raises:
            Exception: For any errors that occur during retrieval.
        """
        try:
            kwargs = kwargs or {}
            args = args or {}

            retrieval_results = self.collection.find(args, kwargs)
            return retrieval_results
        except Exception as ex:
            print(ex)

    def upload_to_db(self, transcript, frames_timestamp):
        combine_docs = self.combine_sentences_with_context(transcript)

        texts = [doc["text"] for doc in combine_docs]
        metadata = [
            {
                "start": doc.get("start", 0.0) if doc["start"] is not None else 0.0,
                "end": doc.get("end", 0.0) if doc["end"] is not None else 0.0
            }
            for doc in combine_docs
        ]

        video_document = {
            "video_url": "",
            "transcript": ''.join(texts),
            "chunk_count": len(texts)
        }
        video_insert_result = self.videos_collection.insert_one(video_document)
        video_id = video_insert_result.inserted_id

        # Build an index
        embedding_function = SentenceTransformerEmbeddings(
            model_name=self.embedding_model
        )
        # Generate and store embeddings
        for i, chunk in enumerate(texts):
            # Combine chunk text with relevant frames' tag information
            relevant_frames = [
                frame for frame in frames_timestamp if frame['timestamp'] >= metadata[i]['start'] and frame['timestamp'] <= metadata[i]['end']
            ]

            # Extract text from the frames' tags
            frame_texts = []
            for frame in relevant_frames:
                if "caption" in frame["tag"] and frame["tag"]["caption"]["text"]:
                    frame_texts.append(frame["tag"]["caption"]["text"])
                if "dense_captions" in frame["tag"] and frame["tag"]["dense_captions"]:
                    frame_texts.extend([caption["text"] for caption in frame["tag"]["dense_captions"]])
                if "read" in frame["tag"] and frame["tag"]["read"]:
                    frame_texts.extend([line["line"] for line in frame["tag"]["read"]])
                if "tags" in frame["tag"] and frame["tag"]["tags"]:
                    frame_texts.extend([line["name"] for line in frame["tag"]["tags"]])
                if "objects" in frame["tag"] and frame["tag"]["objects"]:
                    frame_texts.extend([line["name"] for line in frame["tag"]["objects"]])

            # Combine chunk text and frame tag texts into a single string
            combined_text = chunk + " " + " ".join(frame_texts)

            # Generate a single embedding for the combined text
            embedding = embedding_function.embed_documents([combined_text])

            document = {
                "video_id": video_id,
                "transcript_id": f"chunk_{i}",
                "text": chunk,
                "embedding": embedding,
                "metadata": metadata[i],
                "frames": relevant_frames
            }
            self.collection.insert_one(document)

        print("Documents successfully inserted into Azure Cosmos DB")

    # Function to combine short sentences into context windows
    def combine_sentences_with_context(self, segments, max_chunk_size=1000):
        combined_texts = []
        current_chunk = []
        current_chunk_size = 0
        current_start = None
        current_end = None

        for segment in segments:
            text = segment['text']
            start_time_segment = segment['start']
            text_size = len(text)

            # Start a new chunk if adding this sentence would exceed the size limit
            if current_chunk_size + text_size > max_chunk_size:
                # Save the current chunk
                combined_texts.append({
                    "text": ' '.join(current_chunk),
                    "start": current_start,
                    "end": segment['end']
                })
                # Start a new chunk
                current_chunk = [convert_seconds_to_mm_ss(start_time_segment) + " " + text]
                current_chunk_size = text_size
                current_start = segment['start']
            else:
                # Add sentence to the current chunk
                current_chunk.append(" " + convert_seconds_to_mm_ss(start_time_segment) + text)
                current_chunk_size += text_size
                current_end = segment['end']

        # Add any remaining sentences as the final chunk
        if current_chunk:
            combined_texts.append({
                "text": ' '.join(current_chunk),
                "start": current_start,
                "end": current_end
            })

        return combined_texts

    def insert_video(self, video_document):
        return self.videos_collection.insert_one(video_document)

    def insert_embedding(self, document):
        return self.collection.insert_one(document)


