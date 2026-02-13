#!/usr/bin/env python
# coding: utf-8

# In[11]:


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# =========================
# FastAPI Setup
# =========================
app = FastAPI(
    title="Pneumonia Risk Prediction API",
    version="1.0"
)

# =========================
# Load Model
# =========================
model = joblib.load("servey_model.pkl")

LABELS = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk",
    3: "Severe Pneumonia"
}

# =========================
# Pydantic Schema
# =========================
class SurveyInput(BaseModel):
  

    FeverDuration: str
    FeverLevel: str
    FeverResponse: str

    CoughTime: str
    CoughType: str
    PhlegmStatus: str
    CoughSeverity: str

    HasAbnormalBreathingSound: bool
    BreathingEffort: str
    FeedingAbility: str
    HasChestIndrawing: str

    HasNasalFlaring: bool
    HasCyanosis: bool

    FatigueStatus: bool
    AppetiteStatus: str

    HasWeakCry: bool
    HasSevereRunnyNoseWithBreathingDifficulty: bool

    RecurrentIssues: str
    HeartCondition: str

# =========================
# Prediction Endpoint
# =========================
@app.post("/predict")
def predict(data: SurveyInput):
    # تحويل البيانات إلى DataFrame
    df = pd.DataFrame([data.dict()])

    # عمل prediction
    prediction = int(model.predict(df)[0])

    return {
        "class_id": prediction,
        "result": LABELS[prediction]
    }



# In[ ]:









