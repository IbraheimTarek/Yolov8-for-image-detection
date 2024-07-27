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

# Read the Google Analytics HTML file
with open("google_analytics.html", "r") as f:
    html_code = f.read()

# Apply Streamlit theme background color
st.markdown(f"""
<style>
.google-analytics {{
    background-color: var(--background-color);
}}
</style>
<div class="google-analytics">{html_code}</div>
""", unsafe_allow_html=True)

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
        response = requests.post("http://127.0.0.1:5000/predict", files=files)

        if response.status_code == 200:
            response_data = response.json()
            components_list = response_data.get("components", [])
            image_path = response_data.get("image_path", None)

            if image_path:
                processed_image = Image.open(image_path)
                st.image(processed_image, caption='Processed Image', use_column_width=True)

            if components_list:
                st.success("Components recognized:")
                for component in components_list:
                    st.write(f"{component['label']}: {component['confidence']:.2f}")
            else:
                st.error("No components recognized.")
        else:
            st.error(f"Failed to process the image. Error: {response.text}")
