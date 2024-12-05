import base64
import os
import re

import dotenv
import streamlit as st
import requests

dotenv.load_dotenv()

st.title("Chat with Video")
bytes_data = None

if "file_uploaded" not in st.session_state:
    st.session_state["file_uploaded"] = False

if "video_bytes" not in st.session_state:
    st.session_state["video_bytes"] = None

if "start_time" not in st.session_state:
    st.session_state["start_time"] = 0

if st.session_state["video_bytes"]:
    st.video(st.session_state["video_bytes"], start_time=st.session_state["start_time"])

uploaded_files = st.file_uploader(
    "Choose a MP4 file", accept_multiple_files=True, type=["mp4"]
)

if uploaded_files and not st.session_state["file_uploaded"]:
    file_list = []

    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.session_state["video_bytes"] = bytes_data
        st.write("filename:", uploaded_file.name)
        st.video(bytes_data)

        encoded_data = base64.b64encode(bytes_data).decode('utf-8')

        file_list.append({
            "filename": uploaded_file.name,
            "file_data": encoded_data
        })


    response = requests.post(
        os.environ.get("SERVER_ADDRESS") + "v2/context/video",
        json={"file_list": file_list}
    )

    st.session_state["file_uploaded"] = True

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = requests.post(os.environ.get("SERVER_ADDRESS") + "prompt/video", json={"prompt": prompt})
        response = response.json()["response"]
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        timestamps = re.findall(r"\[(\d{1,2}):(\d{2})", response)
        if timestamps and bytes_data:
            for minutes, seconds in timestamps:
                start_time = int(minutes) * 60 + int(seconds)
                if st.button(f"Jump to {minutes}:{seconds}"):
                    # Display the video with the specified start time
                    st.video(bytes_data, start_time=start_time)
