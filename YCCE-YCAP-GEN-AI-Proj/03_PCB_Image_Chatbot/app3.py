import streamlit as st
from PIL import Image
import ollama

st.set_page_config(page_title="PCB AI Inspection Assistant")

st.title("🤖 PCB AI Inspection Assistant")

uploaded_file = st.file_uploader(
    "Upload PCB Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded PCB", use_container_width=True)

    if st.button("Inspect PCB"):

        with st.spinner("Analyzing PCB..."):

            response = ollama.chat(
                model="gemma2:2b",      # Change to your vision model
                messages=[
                    {
                        "role": "user",
                        "content": """
You are an IPC Certified PCB Quality Inspector.

Analyze this PCB image.

Return:

PCB Status

Detected Components

Detected Defects

Defect Location

Possible Cause

Severity

Recommendation

Confidence

If unsure, clearly mention manual inspection is recommended.
""",
                        "images": [uploaded_file.getvalue()]
                    }
                ]
            )

        st.subheader("Inspection Result")

        st.write(response["message"]["content"])