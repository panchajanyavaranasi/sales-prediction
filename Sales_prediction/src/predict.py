import pickle
import numpy as np

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def predict(model, input_data):
    return model.predict([input_data])