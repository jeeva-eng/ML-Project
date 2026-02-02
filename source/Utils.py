import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from source.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def evaluate_models(X_train, y_train, X_test, y_test, models, param=None):
    try:
        report = {}

        for i, (name, model) in enumerate(models.items()):
            # Only do GridSearch if param is provided
            if param is not None:
                para = param[name]
                from sklearn.model_selection import GridSearchCV
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)

            # Train the model
            model.fit(X_train, y_train)

            # Predictions
            y_test_pred = model.predict(X_test)

            # Evaluate
            test_model_score = r2_score(y_test, y_test_pred)
            report[name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)