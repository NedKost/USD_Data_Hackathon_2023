from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from src.data_preprocessing import preprocessed_data 
import pandas as pd
import joblib

def batch_predict():
    # Create columns for non-Ordinal Features
    label_encoder = LabelEncoder()  

    data = preprocessed_data(is_training=False)

    # Load the preprocessed data
    X = data

    # Load the saved model
    model_filename = 'src/model_inputs_outputs/model/artifacts/trained_xgboost_model.joblib'
    gb_model = joblib.load(model_filename)

    gb_predict = gb_model.predict(X)

    gb_predict_decoded = label_encoder.inverse_transform(gb_predict)
    print(gb_predict_decoded)
    
    
