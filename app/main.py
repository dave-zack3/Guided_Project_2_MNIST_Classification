from fastapi import FastAPI
import mlflow.pyfunc

# Load the MLflow model as a PyFunc model
app = FastAPI()

model = mlflow.pyfunc.load_model('./optmized_model/artifacts/model')

@app.post('/predict')
def predict(data: dict):
    # Assuming data is a dictionary with model input
    
    prediction = model.predict(data)
    return {"prediction": prediction}