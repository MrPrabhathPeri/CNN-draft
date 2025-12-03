import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ------------ CONFIG ------------
MODEL_PATH = "defect_detection_resnet50_final.h5"

# Use the exact order printed in Colab for class_names
CLASS_NAMES = ["def_front", "ok_front"]  # example: update if yours are different

IMG_HEIGHT = 224
IMG_WIDTH = 224
# --------------------------------

# Load model once when the script starts
@st.cache_data(show_spinner=False)
def _load_dummy():
    # tiny cached function just to avoid re-running everything
    return True

_ = _load_dummy()

# we do not cache the Keras model, just load it at module import
model = keras.models.load_model(MODEL_PATH)

st.title("Manufacturing Defect Detection")
st.write("Upload an image of a casting part to classify it as **Defective** or **OK**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Same preprocess as used for training (ResNet50)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            img_batch = preprocess_image(image)
            preds = model.predict(img_batch)
            prob = float(preds[0][0])

            threshold = 0.5
            predicted_label_idx = int(prob > threshold)
            predicted_class = CLASS_NAMES[predicted_label_idx]

            # Adjust these messages based on which class means "defective"
            if predicted_label_idx == 0:
                status_text = f"Prediction: **{predicted_class}** (Defective part)"
            else:
                status_text = f"Prediction: **{predicted_class}** (OK / Non-defective)"

            st.markdown(status_text)
            st.write(f"Raw score (sigmoid output): `{prob:.4f}`")
