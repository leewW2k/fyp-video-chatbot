import base64
import os
import re

import dotenv
import streamlit as st
import requests

dotenv.load_dotenv()

st.set_page_config(layout="wide")
st.title("Video Chatbot")
bytes_data = None

col1, col2 = st.columns([2, 1])

if "videos_response" not in st.session_state:
    # Fetch videos only once and store in session state
    response = requests.get(os.environ.get("SERVER_ADDRESS") + "/videos")
    st.session_state["videos_response"] = response.json()["response"]

response = st.session_state["videos_response"]

# Create a mapping of _id to filenames (for dropdown display)
id_to_file_mapping = {
    entry['_id']: entry['filename'].rsplit('.', 1)[0]
    for entry in response
}
id_to_video_map = {
    entry["_id"]: entry["base64_video"]
    for entry in response
}

# Reverse the mapping to use filenames as dropdown options
file_to_id_mapping = {v: k for k, v in id_to_file_mapping.items()}

with col1:
    # Create a dropdown with filenames
    selected_file = st.selectbox("Select a Video:", list(file_to_id_mapping.keys()))

    # Get the corresponding _id
    if selected_file:
        selected_id = file_to_id_mapping[selected_file]
        st.write(f"Video Selected: {selected_file}")
        video_bytes = base64.b64decode(id_to_video_map[selected_id])
        st.video(video_bytes)

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

with col2:
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
            response = requests.post(
                os.environ.get("SERVER_ADDRESS") + "prompt/video", json={
                    "prompt": prompt,
                    "video_id": selected_id
                }
            )
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
