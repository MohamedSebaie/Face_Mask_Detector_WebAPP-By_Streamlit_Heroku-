import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os


@st.cache(hash_funcs={cv2.dnn_Net: hash})
def load_face_detector():
    prototxt_path = os.path.sep.join(["CAFFEE", "deploy.prototxt"])
    weights_path = os.path.sep.join(["CAFFEE", "res10_300x300_ssd_iter_140000.caffemodel"])
    cnn_net = cv2.dnn.readNet(prototxt_path, weights_path)

    return cnn_net

@st.cache(allow_output_mutation=True)
def load_our_model():
    cnn_model = load_model("face_mask.h5")

    return cnn_model

image= Image.open('imageB.jpg')
st.image(image,use_column_width=True)

st.write("""
    # Face Mask Detector By **Mohamed Sebaie** 
    By Upload Images and it will detect **Mask** or **No Mask**.
    ***
    """)

net = load_face_detector()
model = load_our_model()

uploaded_image = st.sidebar.file_uploader("Choose a JPG, JPEG or PNG file", type=["jpg","jpeg","png"])
confidence_value = st.sidebar.slider('Confidence:', 0.0, 1.0, 0.5, 0.1)
if uploaded_image:
    st.sidebar.info('Uploaded image:')
    st.sidebar.image(uploaded_image, width=240)
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig = image.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_value:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            expanded_face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(expanded_face)[0]

            predicted_class = 0
            label = "No Mask"
            if mask > withoutMask:
                label = "Mask"
                predicted_class = 1

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            st.image(image, width=640)
            st.write('### ' + label)

