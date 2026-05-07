import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("model.h5")
class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

st.title("Scene Classifier")
st.write("Upload an image to predict the scene type")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(200, 200))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    predicted_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    st.success(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")

    st.write("Class probabilities:")
    for cls, prob in zip(class_names, preds):
        st.write(f"- {cls}: {prob*100:.2f}%")