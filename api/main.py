from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf 
MODEL=tf.keras.models.load_model('../Models_Potato/my_model_potato.keras')
CLASS_NAMES = ['Early_blight', 'Late_blight', 'Healthy']
app=FastAPI()


def read_file_as_image(data)->np.array:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.get("/ping")
async def ping():
    return "Hello World"
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image= read_file_as_image(await file.read())
    image_batch=np.expand_dims(image,0)
    predictions=MODEL.predict(image_batch)
    prediction_class=CLASS_NAMES[np.argmax(predictions[0])]
    confidence=np.max(predictions[0])
    return {
        'class': prediction_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)