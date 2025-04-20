import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------- Custom Page Config --------------------
st.set_page_config(
    page_title="🚀 Satellite Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------- Universe-style Header --------------------
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #9be7ff; font-family: Monospace; font-size: 48px;'>🌌 Satellite Image Classifier</h1>
        <h4 style='color: #bdbdbd;'>Explore the Universe from Above ☁️🛰️🌍</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- Load Model --------------------
model = tf.keras.models.load_model('Satelight.keras')

# Example class names (Change according to your model)
class_names = ['Forest', 'Urban', 'Water', 'Agriculture']

# -------------------- Image Upload --------------------
uploaded_file = st.file_uploader("Upload a Satellite Image 🖼️", type=["jpg", "jpeg", "png"])

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.expand_dims(image, axis=0)

# -------------------- Prediction --------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🛰️ Your Uploaded Image", use_column_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Classifying...")
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f"<h3 style='color:#00e676;'>✅ Predicted Class: {predicted_class}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:#ffcc80;'>🔥 Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)

    # Optional: Confidence Bar Chart
    st.markdown("### 📊 Prediction Confidence")
    fig, ax = plt.subplots()
    ax.barh(class_names, prediction, color='#90caf9')
    ax.set_xlim([0, 1])
    for i, v in enumerate(prediction):
        ax.text(v + 0.01, i, f"{v:.2f}", color='white', fontweight='bold')
    st.pyplot(fig)
