# from fastapi import FastAPI, File, UploadFile 
import tensorflow as tf
import json 
from model_definition import SegmentationModel 
import cv2
import json
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# load modle
model = SegmentationModel().model
model.load_weights('cancer_weights.h5') 

#load and show image
img_path = 'C:/Users/shyam.i.albert/Documents/python-models/SegmentationAPI/testimg.jpg'
img = Image.open(img_path)



# # Convert to RGB if the image has an alpha channel (RGBA)
# if img.mode == 'RGBA':
#     img = img.convert('RGB')

# # Resize the image to match the model's expected input size
img = img.resize((256, 256))



# Convert PIL Image to NumPy array and then to TensorFlow tensor
image_array = np.array(img)
# image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)


# # Normalize the image (optional, depending on the model's requirements)
# image_tensor = image_tensor / 255.0

# # Expand dimensions to match model input shape
image_tensor = tf.expand_dims(image_array, axis=0)


# Predict using the model
yhat = model.predict(image_tensor)


prediction = yhat.tolist()
yhat = np.array(prediction)
yhat = np.squeeze(np.where(yhat > 0.3, 1.0, 0.0))

x = cv2.imread('C:/Users/shyam.i.albert/Documents/python-models/SegmentationAPI/testimg.jpg')
# Display the output
fig, ax = plt.subplots(1,7, figsize=(20,10))
ax[0].imshow(x) 
for i in range(6):
    ax[i+1].imshow(yhat[:,:,i])
plt.show()
