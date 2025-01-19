import base64
import json
import os
from datetime import datetime

from azure.storage.blob import BlobServiceClient

import dotenv

import whisper

from azure_blob_service import AzureBlobService
from utils import process_file
from database_service import AzureDatabaseService
from vision_service import VisionService
from utils import convert_seconds_to_mm_ss

dotenv.load_dotenv()


class VideoRetriever:
    def __init__(
            self,
            database_service: AzureDatabaseService,
            storage_service: AzureBlobService,
            vision_service: VisionService,
            video_url: str = None,
            save_dir: str = "./yt_audios",
    ):
        self.video_url = video_url
        self.save_dir = "./yt_audios"
        self.local = False
        self.model_name = "all-MiniLM-L6-v2"
        self.vision_service = vision_service
        self.database_service = database_service
        self.storage_service = storage_service

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

    def process_blob(self, video_data):
        try:
            for video in video_data:
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
                    video_document = {
                        "filename": video.filename,
                        "upload_date": datetime.now(),
                    }
                    video_id = self.database_service.insert_video_entry(video_document).inserted_id
                else:
                    print(f"{video.filename} already exists in Azure Blob Storage. Skipping upload.")
                    video_id = self.database_service.find_video_entry(video.filename)

                local_video_path = os.path.join(self.save_dir, video.filename)
                process_file(fp=local_video_path, mode='wb', content=decoded_bytes)

                # Connect to Azure Blob Storage to upload the transcript
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                container_client = blob_service_client.get_container_client(container_name)
                transcript_blob_name = f"{os.path.splitext(video.filename)[0]}_timetranscript.txt"
                transcript_blob_client = container_client.get_blob_client(transcript_blob_name)
                if not transcript_blob_client.exists():
                    model = whisper.load_model("base")
                    result = model.transcribe(local_video_path, word_timestamps=True)

                    transcript_text = result['segments']
                    transcript_json = json.dumps(transcript_text, ensure_ascii=False, indent=4)

                    transcript_blob_client.upload_blob(transcript_json, overwrite=True)
                else:
                    print(f"{transcript_blob_name} already exists in Azure Blob Storage. Skipping upload.")
                    transcript_text = self.storage_service.fetch_transcript_from_blob(transcript_blob_name)

                timestamp_frames = self.vision_service.extract_relevant_frames(video_path=local_video_path, output_dir="frames")

                for timestamp_frame in timestamp_frames:
                    image_data = process_file(fp=timestamp_frame["frame_path"], mode='rb')
                    output = self.vision_service.analyze_image(image_data=image_data)
                    timestamp_frame["tag"] = output

                self.database_service.upload_to_db(
                    transcript=transcript_text,
                    frames_timestamp=timestamp_frames,
                    video_reference_id=video_id,
                )
        except Exception as e:
            print(e)
