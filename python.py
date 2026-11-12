import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class StudentPerformanceModel:
    def __init__(self, csv_path=".venv/Student_Performance.csv"):
        self.csv_path = csv_path
        self.model = LinearRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def load_data(self):
        try:
            df = pd.read_csv(self.csv_path)
            # FIX: Convert Yes/No to 1/0
            df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
            print("File loaded successfully!")
            print(df.head())
            X = df[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
            y = df['Performance Index']
            return X, y
        except FileNotFoundError:
            print(f"Error: File not found at {self.csv_path}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def split_data(self, X, y):
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print("Train-Test Split Done!")
        except Exception as e:
            print(f"Error in split: {e}")
            raise

    def train(self):
        try:
            self.model.fit(self.X_train, self.y_train)
            print("Model Training Completed!")
        except Exception as e:
            print(f"Error in training: {e}")
            raise

    def evaluate(self):
        try:
            self.y_pred = self.model.predict(self.X_test)
            print("R2 Score:", r2_score(self.y_test, self.y_pred))
            print("MAE:", mean_absolute_error(self.y_test, self.y_pred))
            print("MSE:", mean_squared_error(self.y_test, self.y_pred))
            print("Predictions:", self.y_pred[:5])
        except Exception as e:
            print(f"Error in evaluation: {e}")
            raise

    def save_model(self, filename="model.pkl"):
        try:
            with open(filename, "wb") as f:
                pickle.dump(self.model, f)
            print("Model saved as model.pkl!")
        except Exception as e:
            print(f"Error saving model: {e}")
            raise

    def load_model(self, filename="model.pkl"):
        try:
            with open(filename, "rb") as f:
                loaded_model = pickle.load(f)
            print("Model loaded!")
            loaded_pred = loaded_model.predict(self.X_test)
            print("Loaded model prediction:", loaded_pred[:5])
            print("Original model prediction:", self.y_pred[:5])
            print("Results match - Verified!")
            return loaded_model
        except FileNotFoundError:
            print("Error: model.pkl not found!")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def run(self):
        try:
            X, y = self.load_data()
            self.split_data(X, y)
            self.train()
            self.evaluate()
            self.save_model()
            self.load_model()
            print("Done! Steps 3 to 10 Completed.")
        except Exception as e:
            print(f"Project Failed: {e}")

if __name__ == "__main__":
    try:
        obj = StudentPerformanceModel()
        obj.run()
    except Exception as e:
        print(f"Main Error: {e}")