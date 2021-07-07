import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import imageio
import cv2
import os
from tensorflow.keras.models import model_from_json
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from os import listdir
from gtts import gTTS
from playsound import playsound

def function(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),2)
  th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
  ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  return res

d = {0: 'A', 1: 'B', 2: 'C', 3: 'D',
    4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'}

json_file = open('model-bw.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

upper_left = (335, 55)
bottom_right = (635, 355)

cam = cv2.VideoCapture(0)
cv2.namedWindow("test", cv2.WINDOW_NORMAL)
cv2.resizeWindow('test', 1050,1250)

a=""
g=""
i=1
while True:
    ret, frame = cam.read()
    frame= cv2.flip(frame, 1)
    r = cv2.rectangle(frame, upper_left, bottom_right, (0, 0, 0), 5)
    rect_img = frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    sketcher_rect = rect_img
    sketcher_rect = function(sketcher_rect)
    sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)
    frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]] = sketcher_rect_rgb
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print(g)
        tts = gTTS(g,lang='en') #Provide the string to convert to speech
        m=str(i)+".mp3"
        tts.save(m) #save the string converted to speech as a .wav file
        playsound(m)
        i=i+1
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "frame_0.png"
        print(sketcher_rect.shape)
        cv2.imwrite(img_name, sketcher_rect)
        print("{} written!".format(img_name))
        x = image.img_to_array(sketcher_rect)
        x = np.expand_dims(x, axis=0)
        x = x/255.0
        pre = loaded_model.predict(x)
        p_test=np.argmax(pre)
        a = d[p_test]
        g=g+a
        print(a)
        tts = gTTS(a,lang='en') #Provide the string to convert to speech
        m=str(i)+".mp3"
        tts.save(m) #save the string converted to speech as a .wav file
        playsound(m)
        i=i+1
        # Audio(sound_file, autoplay=True)
cam.release()

cv2.destroyAllWindows()

