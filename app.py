'''from fastapi import FastAPI
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
    return {"prediction": prediction.tolist()}'''

'''from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Example route
@app.route('/')
def home():
    return jsonify({"message": "API is running!"})

# Example predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Add your prediction logic here
    return jsonify({"prediction": "sample output"})

if __name__ == "__main__":
    # Get port from Render's environment or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Run app on all IPs to avoid binding issues
    app.run(host="0.0.0.0", port=port)'''


from fastapi import FastAPI, Request
from pydantic import BaseModel
import os

app = FastAPI()

# Define request body model
class PredictRequest(BaseModel):
    input_data: str

# Example route
@app.get("/")
def home():
    return {"message": "API is running!"}

# Example predict route
@app.post("/predict")
def predict(request: PredictRequest):
    # Add your prediction logic here
    return {"prediction": f"Received: {request.input_data}"}

if __name__ == "__main__":
    # Get port from Render's environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    # Run app on all IPs to avoid binding issues
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)


