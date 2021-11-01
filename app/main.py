from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel
from keras.models import load_model
from typing import List
import io
import numpy as np
import sys

labels = ['Tomato___healthy',
 'Tomato___Late_blight',
 'Apple___Black_rot',
 'Blueberry___healthy',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Leaf_Mold',
 'Corn_(maize)___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Apple___Cedar_apple_rust',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Peach___Bacterial_spot',
 'Cherry_(including_sour)___Powdery_mildew',
 'Corn_(maize)___Common_rust_',
 'Raspberry___healthy',
 'Apple___healthy',
 'Cherry_(including_sour)___healthy',
 'Grape___Esca_(Black_Measles)',
 'Strawberry___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Apple___Apple_scab',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Grape___Black_rot',
 'Tomato___Target_Spot',
 'Tomato___Bacterial_spot',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Peach___healthy',
 'Potato___Late_blight',
 'Potato___healthy',
 'Grape___healthy',
 'Strawberry___Leaf_scorch',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Early_blight',
 'Pepper,_bell___Bacterial_spot',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)']


# Load the model
filepath = './app/final_vgg.hdf5'
model = load_model(filepath)

# Get the input shape for the model layer
input_shape = model.layers[0].input_shape

# Define the FastAPI app
app = FastAPI()

# Define the Response
class Prediction(BaseModel):
  filename: str
  contenttype: str
  prediction: List[float] = []
  likely_class: int
  label: str

# Define the main route
@app.get('/')
def root_route():
  return { 'error': 'Use POST /prediction instead of the root route!' }

# Define the /prediction route
@app.post('/prediction/', response_model=Prediction)
async def prediction_route(file: UploadFile = File(...)):

  # Ensure that this is an image
  if file.content_type.startswith('image/') is False:
    raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

  try:
    # Read image contents
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))

    # Resize image to expected input shape
    pil_image = pil_image.resize((input_shape[1], input_shape[2]))

    # Convert from RGBA to RGB *to avoid alpha channels*
    if pil_image.mode == 'RGBA':
      pil_image = pil_image.convert('RGB')

    # Convert image into grayscale *if expected*
    if input_shape[3] and input_shape[3] == 1:
      pil_image = pil_image.convert('L')

    # Convert image into numpy format
    numpy_image = np.array(pil_image).reshape((input_shape[1], input_shape[2], input_shape[3]))

    # Scale data (depending on your model)
    numpy_image = numpy_image / 255

    # Generate prediction
    prediction_array = np.array([numpy_image])
    predictions = model.predict(prediction_array)
    prediction = predictions[0]
    likely_class = np.argmax(prediction)

    return {
      'filename': file.filename,
      'contenttype': file.content_type,
      'prediction': prediction.tolist(),
      'likely_class': likely_class,
      'label': labels[likely_class]
    }
  except:
    e = sys.exc_info()[1]
    raise HTTPException(status_code=500, detail=str(e))
