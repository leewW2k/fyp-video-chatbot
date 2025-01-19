import os
from typing import List

from fastapi import FastAPI

from azure_blob_service import AzureBlobService
from utils import process_file
from vision_service import VisionService
from database_service import AzureDatabaseService
from openai_service import OpenAIService
from retriever import VideoRetriever
from pydantic import BaseModel

database_service = AzureDatabaseService(
    mongo_connection_string=os.environ.get("MONGODB_CONNECTION_STRING"),
    database_name="video_transcripts"
)
service = OpenAIService(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    deployment_name=os.environ.get("YOUR_DEPLOYMENT_NAME"),
    api_version=os.environ.get("OPENAI_API_VERSION"),
    database_service=database_service
)
azure_blob_service = AzureBlobService(
    azure_storage_connection_string=os.environ.get("AZURE_STORAGE_CONNECTION"),
    azure_container_name=os.environ.get("CONTAINER_NAME"),
)
vision_service = VisionService(
    vision_endpoint=os.environ.get("VISION_ENDPOINT"),
    vision_key=os.environ.get("VISION_KEY")
)
video_retriever = VideoRetriever(
    storage_service=azure_blob_service,
    database_service=database_service,
    vision_service=vision_service
)
app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    video_id: str

class Video(BaseModel):
    video_url: str

class VideoData(BaseModel):
    filename: str
    file_data: str

class VideoList(BaseModel):
    file_list: List[VideoData]


@app.post("/prompt/video")
async def prompt_chatbot(request: PromptRequest):
    try:
        response = await service.generate_video_prompt_response(request.prompt, request.video_id)
        return {"response": response}
    except Exception as ex:
        return ex

@app.post("/v2/context/video")
async def create_video_context(request: VideoList):
    try:
        video_retriever.process_blob(video_data=request.file_list)
        return {"response": "completed"}
    except Exception as ex:
        return ex

@app.get("/videos")
async def get_videos():
    try:
        response = database_service.get_video_list()
        for video in response:
            video["base64_video"] = azure_blob_service.fetch_video_from_blob(video["filename"])
        return {"response": response}
    except Exception as ex:
        return ex
