from source.components.data_transformation import DataTransformation
from source.components.model_trainer import ModelTrainer
import os
os.makedirs("artifacts", exist_ok=True)
# Step 1: Transform Data
data_transformation = DataTransformation()
train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
    train_path="artifacts/train.csv",
    test_path="artifacts/test.csv"
)
print("Data transformation done ✅")

# Step 2: Train Model
model_trainer = ModelTrainer()
r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)
print(f"Best model R2 score: {r2}")
print("Model training done ✅")

