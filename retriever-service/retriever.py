import os
import dotenv

from langchain_community.document_loaders.generic import GenericLoader
from azure_openai_whisper_parser import AzureOpenAIWhisperParser
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser, OpenAIWhisperParserLocal
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Directory to save audio files
save_dir = "./YouTube"

dotenv.load_dotenv()

try:
    # get url
    url = ["https://youtube.com/shorts/IicbiwTAslE?si=H1qA7---M4ZiuHTc"]
    # Transcribe the videos to text
    loader = GenericLoader(
        YoutubeAudioLoader(url, save_dir), AzureOpenAIWhisperParser(
            os.environ.get("WHISPER_API_KEY"),
            os.environ.get("WHISPER_ENDPOINT")
        )
    )
    docs = loader.load()

    # Combine docs
    combined_docs = [doc.page_content for doc in docs]
    text = " ".join(combined_docs)

    # Split the combined docs into chunks of size 1500 with an overlap of 150
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=150
    )
    splits = text_splitter.split_text(text)

    # Build an index
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectordb = Chroma.from_texts(
        splits, embedding_function,
        persist_directory="vector_store_0003",
        collection_name="david_goggins_short"
    )


except Exception as e:
    print(e)
