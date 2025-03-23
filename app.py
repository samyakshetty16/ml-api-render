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


'''from fastapi import FastAPI, Request
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
    uvicorn.run(app, host="0.0.0.0", port=port)'''

from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI()

# Define request body model
class PredictRequest(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: int
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

# Example route
@app.get("/")
def home():
    return {"message": "API is running!"}

# Predict route
@app.post("/predict")
def predict(request: PredictRequest):
    # Example loan eligibility logic
    if (
        request.person_income > 50000
        and request.person_emp_length > 2
        and request.loan_percent_income < 0.3
        and request.cb_person_default_on_file == "N"
    ):
        result = "Eligible for loan"
    else:
        result = "Not eligible for loan"

    return {"prediction": result}

if __name__ == "__main__":
    # Get port from Render's environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    # Run app on all IPs to avoid binding issues
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)



