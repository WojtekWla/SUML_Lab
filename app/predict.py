import joblib
import numpy as np

class IrisModel:
    def __init__(self, path_to_model):
        self.model = joblib.load(path_to_model)
        self.result_map = {
            0: "Iris-setosa",
            1: "Iris-versicolor",
            2: "Iris-virginica"
        }

    def predict(self, sepal_length, sepal_width, petal_length, petal_width):
        pred = self.model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))
        return self.result_map[np.argmax(pred[0])]
