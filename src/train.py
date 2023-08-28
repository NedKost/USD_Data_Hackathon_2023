from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import warnings

from src.data_preprocessing import preprocessed_data 

warnings.filterwarnings("ignore")


def train_model():
    cat_lbl = ['drunk_driver_involved', 'speeding_driver_involved', 'other']

    # Create columns for non-Ordinal Features
    label_encoder = LabelEncoder()  

    data = preprocessed_data(is_training=True)

    # Load the preprocessed data
    X = data[0]
    y = label_encoder.fit_transform(data[1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=500)

    sample_weights = compute_sample_weight(class_weight='balanced', y=y)
        
    gb_model = XGBClassifier(n_estimators=190, max_depth=6, learning_rate=0.13465232824084064, random_state=42)
    gb_model.fit(X, y, sample_weight=sample_weights)

    # Save the trained model using joblib
    model_filename = 'src/model_inputs_outputs/model/artifacts/trained_xgboost_model.joblib'
    joblib.dump(gb_model, model_filename)