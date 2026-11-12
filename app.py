from flask import Flask, render_template, request
import pickle
import numpy as np
import os

class ModelLoader:
    def __init__(self):
        self.model = None
        self.load()

    def load(self):
        try:
            if os.path.exists("model.pkl"):
                path = "model.pkl"
            else:
                path = ".venv/model.pkl"
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            print("Model loaded for website!")
        except FileNotFoundError:
            print("Error: model.pkl not found!")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict(self, data):
        try:
            result = self.model.predict(data)[0]
            return round(float(result), 2)
        except Exception as e:
            print(f"Prediction Error: {e}")
            raise

class FlaskApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.loader = ModelLoader()
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/', methods=['GET', 'POST'])
        def home():
            prediction = None
            try:
                if request.method == 'POST':
                    hours = float(request.form['hours'])
                    previous = float(request.form['previous'])
                    extra = int(request.form['extra'])
                    sleep = float(request.form['sleep'])
                    papers = float(request.form['papers'])
                    data = np.array([[hours, previous, extra, sleep, papers]])
                    prediction = self.loader.predict(data)
            except ValueError:
                prediction = "Please enter valid numbers!"
            except Exception as e:
                print(f"Route Error: {e}")
                prediction = "Error in prediction!"
            return render_template('index.html', prediction=prediction)

    def run(self):
        try:
            self.app.run(debug=True)
        except Exception as e:
            print(f"Flask App Error: {e}")

# For Render / Gunicorn - it needs 'app' variable
try:
    flask_obj = FlaskApp()
    app = flask_obj.app
except Exception as e:
    print(f"Gunicorn Load Error: {e}")

if __name__ == '__main__':
    try:
        flask_obj.run()
    except Exception as e:
        print(f"Main Error: {e}")