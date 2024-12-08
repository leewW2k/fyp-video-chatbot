import base64
import json
import os
import shutil

import numpy as np
import pymongo
from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser, OpenAIWhisperParserLocal
from langchain_community.document_loaders import YoutubeAudioLoader

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings

import dotenv

import whisper
from sklearn.metrics.pairwise import cosine_similarity
from vision_service import VisionService
from utils import convert_seconds_to_mm_ss

dotenv.load_dotenv()


class VideoRetriever:
    def __init__(self, video_url: str = None, video_data = None):
        self.video_url = video_url
        self.video_data = video_data
        self.save_dir = "./yt_audios"
        self.local = True
        self.model_name = "all-MiniLM-L6-v2"
        mongo_client = pymongo.MongoClient(os.environ.get("MONGODB_CONNECTION_STRING"))
        db = mongo_client["video_transcripts"]
        self.collection = db["transcripts"]
        self.videos_collection = db["videos"]
        self.vision_service = VisionService()

    def retrieve_video(self):
        try:
            url = [self.video_url]
            if self.local:
                loader = GenericLoader(
                    YoutubeAudioLoader(url, self.save_dir), OpenAIWhisperParserLocal()
                )
            else:
                loader = GenericLoader(YoutubeAudioLoader(
                    urls=url,
                    save_dir=self.save_dir
                ), OpenAIWhisperParser())

            docs = loader.load()

            model = whisper.load_model("base")
            files = [f for f in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, f))]
            audio = whisper.load_audio(self.save_dir + "/" + files[0])
            result = model.transcribe(audio)

            combine_docs = self.combine_sentences_with_context(result["segments"])

            texts = [doc["text"] for doc in combine_docs]
            metadata = [
                {
                    "start": doc.get("start", 0.0) if doc["start"] is not None else 0.0,
                    "end": doc.get("end", 0.0) if doc["end"] is not None else 0.0
                }
                for doc in combine_docs
            ]

            video_document = {
                "video_url": url,
                "transcript": ''.join(texts),
                "chunk_count": len(texts)
            }
            video_insert_result = self.videos_collection.insert_one(video_document)
            video_id = video_insert_result.inserted_id

            # Build an index
            embedding_function = SentenceTransformerEmbeddings(
                model_name=self.model_name
            )
            # Generate and store embeddings
            for i, chunk in enumerate(texts):
                embedding = embedding_function.embed_documents([chunk])
                document = {
                    "video_id": video_id,
                    "transcript_id": f"chunk_{i}",
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": metadata[i]
                }
                self.collection.insert_one(document)

            print("Documents successfully inserted into Azure Cosmos DB")
        except Exception as e:
            print(e)

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

    def process_blob(self):
        try:
            for video in self.video_data:
                connection_string = os.environ.get("AZURE_STORAGE_CONNECTION")
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                container_name = os.environ.get("CONTAINER_NAME")
                container_client = blob_service_client.get_container_client(container_name)
                try:
                    container_client.create_container()
                except Exception as e:
                    print(e)
                    pass
                decoded_bytes = base64.b64decode(video.file_data)
                blob_client = container_client.get_blob_client(video.filename)
                if not blob_client.exists():
                    # Upload the blob only if it does not exist
                    blob_client.upload_blob(decoded_bytes, overwrite=True)
                    print(f"Uploaded {video.filename} to Azure Blob Storage.")
                else:
                    print(f"{video.filename} already exists in Azure Blob Storage. Skipping upload.")

                local_video_path = os.path.join(self.save_dir, video.filename)
                with open(local_video_path, 'wb') as video_file:
                    video_file.write(decoded_bytes)

                timestamp_frames = self.vision_service.extract_relevant_frames(video_path=local_video_path, output_dir="frames")

                # Connect to Azure Blob Storage to upload the transcript
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                container_client = blob_service_client.get_container_client(container_name)
                transcript_blob_name = f"{os.path.splitext(video.filename)[0]}_timetranscript.txt"
                transcript_blob_client = container_client.get_blob_client(transcript_blob_name)
                transcript_text = None
                if not transcript_blob_client.exists():
                    model = whisper.load_model("base")
                    result = model.transcribe(local_video_path, word_timestamps=True)

                    transcript_text = result['segments']
                    transcript_json = json.dumps(transcript_text, ensure_ascii=False, indent=4)

                    transcript_blob_client.upload_blob(transcript_json, overwrite=True)
                else:
                    print(f"{transcript_blob_name} already exists in Azure Blob Storage. Skipping upload.")
                    transcript_text = self.fetch_transcript_from_blob(transcript_blob_name)


                for timestamp_frame in timestamp_frames:
                    with open(timestamp_frame["frame_path"], "rb") as f:
                        image_data = f.read()
                    output = self.vision_service.analyze_image(image_data=image_data)
                    timestamp_frame["tag"] = output
                print("completed")

                self.upload_to_db(transcript=transcript_text, frames_timestamp=timestamp_frames)
        except Exception as e:
            print(e)


    def fetch_transcript_from_blob(self, blob_name):
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_name = os.environ.get("CONTAINER_NAME")
        container_client = blob_service_client.get_container_client(container_name)

        blob_client = container_client.get_blob_client(blob_name)
        if blob_client.exists():
            blob_content = blob_client.download_blob().readall()
            transcript_json = blob_content.decode('utf-8')
            transcript_list = json.loads(transcript_json)
            return transcript_list
        else:
            print(f"{blob_name} does not exist in Azure Blob Storage.")
            return None

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
            model_name=self.model_name
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
