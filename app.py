from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model("loan_approval_nn_model.keras")

# Define the request format
class InputData(BaseModel):
    input: list  # Adjust based on your model’s input format

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([data.input])
    prediction = model.predict(input_array)
    return {"prediction": prediction.tolist()}
