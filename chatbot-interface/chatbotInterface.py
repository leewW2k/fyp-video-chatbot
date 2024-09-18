import os

import dotenv
import streamlit as st
import requests

dotenv.load_dotenv()

st.title("Chat with Video")

video_form = st.form("my_form")
video_url = video_form.text_input('Video URL:', 'https://www.youtube.com/shorts/R5IB1ugud1Q')
submitted = video_form.form_submit_button("Submit")
if submitted:
    with st.spinner('Processing...'):
        try:
            # Make the POST request and wait for the response
            response = requests.post(
                os.environ.get("SERVER_ADDRESS") + "context/video",
                json={"video_url": video_url}
            )

            # Handle response
            if response.status_code == 200:
                st.success("Video processed successfully!")
            else:
                st.error("Failed to process the video.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

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


