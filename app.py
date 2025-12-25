import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas



# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_cnn():
    model = load_model("emnist_letters_cnn3.keras")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = load_cnn()

# -------------------- PREPROCESS --------------------
def preprocess_canvas(img):
    # RGBA → Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img = cv2.bitwise_not(img)
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = img[y_min:y_max+1, x_min:x_max+1]
    h, w = cropped.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = cropped
    img = cv2.resize(square, (28, 28))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  
    img = np.expand_dims(img, axis=0)   

    return img

# -------------------- STREAMLIT UI --------------------
st.title("✍️ Handwritten Letter Recognition")
st.write("Draw an **uppercase letter** in the box below")

canvas = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=8,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)
# -------------------- PREDICTION --------------------
if st.button("Predict"):
    if canvas.image_data is not None:
        img = preprocess_canvas(canvas.image_data)
        if img is not None:
            pred = model.predict(img, verbose=0)
            idx = np.argmax(pred)
            letter = chr(idx + ord("A"))
            confidence = pred[0][idx] * 100
            st.subheader(f"Prediction: **{letter}**")
            st.write(f"Confidence: **{confidence:.1f}%**")
        else:
            st.warning("Please draw a letter first!")
