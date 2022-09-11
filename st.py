import streamlit as st
import cv2
import numpy as np
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras. preprocessing import image
model=tf.keras.models.load_model("C:\\Users\\Admin\\Downloads\\Agni_2034003_model (1).h5")
cap = cv2.VideoCapture(0)
FRAME=st.image([])
font = cv2.FONT_HERSHEY_SIMPLEX
org = (0, 30)
fontScale = 0.5
color = (255, 0, 0)
thickness = 2
        

        
        
# loop runs if capturing has been initialized.
while (True):
    # reads frames from a camera
    # ret checks return at each frame
    _, frame = cap.read()
    time.sleep(0.1)
    _, frame1 = cap.read()



    image_1_resize = cv2.resize(frame, (256, 256))
    img = cv2.cvtColor(image_1_resize, cv2.COLOR_BGR2GRAY)
    imgstack = np.dstack([img] * 3)

    image_1_resize1 = cv2.resize(frame1, (256, 256))
    img1 = cv2.cvtColor(image_1_resize1, cv2.COLOR_BGR2GRAY)
    imgstack1=np.dstack([img1]*3)
    absdiff = cv2.absdiff(imgstack, imgstack1)
    FRAME.image(absdiff)
    abs1=np.expand_dims(absdiff,axis=0)
    
    val = model.predict(abs1)
    if val == 0:
        absdiff = cv2.putText(absdiff, 'Unsigned', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        FRAME.image(absdiff)

    else:
        absdiff = cv2.putText(absdiff, 'Signed', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        FRAME.image(absdiff)
    




