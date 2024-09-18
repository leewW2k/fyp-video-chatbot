import json
import os
import shutil

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser, OpenAIWhisperParserLocal
from langchain_community.document_loaders import YoutubeAudioLoader

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings

import dotenv

import whisper

dotenv.load_dotenv()


class VideoRetriever:
    def __init__(self, video_url: str):
        self.video_url = video_url
        self.save_dir = "./yt_audios"
        self.local = True
        self.model_name = "all-MiniLM-L6-v2"
        self.persist_directory = "vector_store_003"
        self.collection_name = "test_video"

    def flushFolder(self, file_directory):
        for filename in os.listdir(file_directory):
            file_path = os.path.join(file_directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def retrieve_video(self):
        self.flushFolder(self.save_dir)
        self.flushFolder(self.persist_directory)
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

            # Build embedding and index
            embedding_function = SentenceTransformerEmbeddings(
                model_name=self.model_name
            )

            Chroma.from_texts(
                texts,
                embedding=embedding_function,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                metadatas=metadata
            )

        except Exception as e:
            print(e)

    # Function to combine short sentences into context windows
    def combine_sentences_with_context(self, segments, max_chunk_size=128):
        combined_texts = []
        current_chunk = []
        current_chunk_size = 0
        current_start = None
        current_end = None

        for segment in segments:
            text = segment['text']
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
                current_chunk = [text]
                current_chunk_size = text_size
                current_start = segment['start']
            else:
                # Add sentence to the current chunk
                current_chunk.append(text)
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
