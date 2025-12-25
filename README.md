# letter-recgoniztion
✍️ Handwritten Letter Recognition (CNN + Streamlit)

This project is an end-to-end handwritten uppercase letter recognition system built using Deep Learning (CNN) and deployed as an interactive web application using Streamlit.

Users can draw a letter directly on a canvas, and the model predicts the corresponding uppercase English alphabet (A–Z) in real time.

 Features

 Convolutional Neural Network (CNN) trained on the EMNIST Letters dataset

 Interactive drawing canvas using Streamlit

 Real-time prediction with confidence score

 Careful image preprocessing (cropping, resizing, normalization)

 Deployed on Streamlit Cloud

 Uses OpenCV for image processing (headless for cloud compatibility)

🛠️ Tech Stack

Python

TensorFlow / Keras

Convolutional Neural Networks (CNN)

OpenCV

Streamlit

NumPy

 How It Works

User draws an uppercase letter on the canvas

The image is:

Converted to grayscale

Inverted to match training data

Cropped using bounding box

Resized to 28×28

The processed image is passed to the CNN

The model outputs:

Predicted letter

Confidence score

📊 Model Performance

Trained on EMNIST Letters

Test accuracy: ~89%

Focused on real-world inference consistency, not just dataset accuracy

▶️ Run Locally
pip install -r requirements.txt
streamlit run app.py

🌐 Live Demo

🔗 Live App on Streamlit Cloud

(replace with your actual link)

📌 Key Learnings

High test accuracy does not guarantee real-world performance

Matching training and inference preprocessing is critical

Deployment environments require headless OpenCV

Building full pipelines teaches more than isolated models

📬 Future Improvements

Support for full words / sentences

Sequence models (CNN + RNN / Transformer)

Mobile-friendly UI

Dataset augmentation for robustness

👤 Author

Sachit Agarwal
Aspiring Machine Learning Engineer

