import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from source.exception import CustomException
from source.logger import logging
from source.Utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Model R2 score: {r2}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            return r2

        except Exception as e:
            raise CustomException(e, sys)
