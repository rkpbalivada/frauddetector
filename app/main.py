from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("app/model.pkl")

@app.get("/")
def root():
    return {"message": "Fraud Detection API"}

@app.post("/predict/")
def predict(data: list):
    try:
        prediction = model.predict([np.array(data)])
        return {"fraud": bool(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

