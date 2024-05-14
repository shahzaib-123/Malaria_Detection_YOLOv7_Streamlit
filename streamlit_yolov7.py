import singleinference_yolov7
from singleinference_yolov7 import SingleInference_YOLOV7
import os
import streamlit as st
import logging
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

class Streamlit_YOLOV7(SingleInference_YOLOV7):

    def __init__(self):
        self.logging_main = logging
        self.logging_main.basicConfig(level=logging.DEBUG)

    def new_yolo_model(self, img_size, path_yolov7_weights, path_img_i, device_i='cpu'):
        super().__init__(img_size, path_yolov7_weights, path_img_i, device_i=device_i)

    def main(self):
        st.title('MALARIA INFECTED CELLS DETECTION USING YOLOV7')
        st.subheader("Upload an image and run YoloV7 on it to detect malaria infected cells. This model was trained to detect the following classes:")
        
        text_i_list = [f'{i}: {name}\n' for i, name in enumerate(self.names)]
        st.selectbox('Classes', tuple(text_i_list))

        st.write("Notice where the model fails (i.e., cells are too close up or too far away).")     
        
        self.response = requests.get(self.path_img_i)
        self.img_screen = Image.open(BytesIO(self.response.content))
        st.image(self.img_screen, caption=self.capt, width=None, use_column_width=True)

        self.im0 = np.array(self.img_screen.convert('RGB'))
        self.load_image_st()
        predictions = st.button('Predict on the image?')
        if predictions:
            self.predict()

    def load_image_st(self):
        uploaded_img = st.file_uploader('Upload an image')
        if uploaded_img is not None:
            self.img_data = uploaded_img.getvalue()
            st.image(self.img_data)
            self.im0 = Image.open(BytesIO(self.img_data))
            self.im0 = np.array(self.im0)

    def predict(self):
        st.write('Loading image...')
        self.load_cv2mat(self.im0)
        st.write('Making inference...')
        self.inference()

        self.img_screen = Image.fromarray(self.image).convert('RGB')
        self.capt = 'DETECTED:'
        if self.predicted_bboxes_PascalVOC:
            for item in self.predicted_bboxes_PascalVOC:
                name = str(item[0])
                conf = str(round(100 * item[-1], 2))
                self.capt += f' name={name} confidence={conf}%, '
        st.image(self.img_screen, caption=self.capt, width=None, use_column_width=True)

if __name__ == '__main__':
    app = Streamlit_YOLOV7()
    img_size = 1056
    path_yolov7_weights = "weights/best.pt"
    path_img_i = "https://raw.githubusercontent.com/shahzaib-123/Malaria_Detection_YOLOv7_Streamlit/main/ini_image.jpg"
    app.capt = "MALARIA"
    app.new_yolo_model(img_size, path_yolov7_weights, path_img_i)
    app.load_model()  # Loading the YOLOv7 model
    app.main()

