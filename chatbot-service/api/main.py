from functools import lru_cache
from typing import Any, List

from dotenv import load_dotenv
from fastapi import FastAPI
from openai_service import OpenAIService
from retriever import VideoRetriever
from . import config
from pydantic import BaseModel

service = OpenAIService()
app = FastAPI()

@lru_cache
def get_settings():
    return config.Settings()


class PromptRequest(BaseModel):
    prompt: str

class Video(BaseModel):
    video_url: str

class VideoData(BaseModel):
    filename: str
    file_data: str

class VideoList(BaseModel):
    file_list: List[VideoData]


@app.post("/prompt")
async def prompt_chatbot(request: PromptRequest):
    try:
        response = await service.create_response(request.prompt)
        return {"response": response}
    except Exception as ex:
        return ex


@app.post("/prompt/video")
async def prompt_chatbot(request: PromptRequest):
    try:
        response = await service.generate_video_prompt_response(request.prompt)
        return {"response": response}
    except Exception as ex:
        return ex


@app.post("/context/video")
async def create_video_context(request: Video):
    try:
        retrieval_service = VideoRetriever(video_url=request.video_url)
        retrieval_service.retrieve_video()
        return {"response": "completed"}
    except Exception as ex:
        return ex

@app.post("/v2/context/video")
async def create_video_context(request: VideoList):
    try:
        retrieval_service = VideoRetriever(video_data=request.file_list)
        retrieval_service.process_blob()
        return {"response": "completed"}
    except Exception as ex:
        return ex