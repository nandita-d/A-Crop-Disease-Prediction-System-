from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model

app = FastAPI()
model = tf.keras.models.load_model('Newfolder/model.keras')
CLASS_NAMES = ["Common_rust", "Gray_Leaf_Spot", "Healthy","Northern_Leaf_Blight","Early_Blight,"]

@app.get('/ping')
async def ping():
    return {"message": "Hello, I'm working"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0) 
        predictions = model.predict(img_batch)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        
        confidence = np.max(predictions[0])
        
        # Return both class prediction and file information
        return {
            "filename": file.filename, 
            "prediction": predicted_class_name, 
            "confidence": float(confidence)
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8003)
