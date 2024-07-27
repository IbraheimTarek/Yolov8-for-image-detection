import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import io
import requests

# Function to load COCO class names
def load_coco_names(file_path):
    with open(file_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# Include Google Analytics tracking code
with open("google_analytics.html", "r") as f:
    html_code = f.read()
    components.html(html_code, height=0)

# Streamlit application
st.title("Image Component Recognition with Bounding Boxes")

# Display COCO class names in a collapsible section
coco_names = load_coco_names("coco.names")
with st.expander("View COCO Classes"):
    st.write(", ".join(coco_names))

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the uploaded image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Button to recognize components
    if st.button("Recognize Components"):
        st.write("Analyzing the image...")
        files = {'file': ('image.png', img_bytes, 'image/png')}
        response = requests.post("https://ibraheimtarek.pythonanywhere.com/predict", files=files)

        if response.status_code == 200:
            processed_image = Image.open(io.BytesIO(response.content))
            st.image(processed_image, caption='Processed Image', use_column_width=True)
        else:
            st.error(f"Failed to process the image. Error: {response.json()['error']}")
