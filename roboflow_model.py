from roboflow import Roboflow

API_KEY = "7C8G3Zmg1cqNo5jeZndY"
WORKSPACE = "vision-bot"
PROJECT = "clash-royale-card-detection-ylzsc"
VERSION = 4

CONFIDENCE = 40
OVERLAP = 30

def load_model():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    return project.version(VERSION).model

def predict_on_path(model, path):
    return model.predict(path, confidence=CONFIDENCE, overlap=OVERLAP).json()