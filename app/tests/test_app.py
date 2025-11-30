import app.predict as p
import unittest

class Tests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Tests, self).__init__(*args, **kwargs)
        self.model = p.IrisModel('../model.joblib')

    def test_predict_iris_setosa(self):
        result = self.model.predict(5.1, 3.5, 1.4, 0.2)
        print(result)
        self.assertEqual(result, "Iris-setosa")

    def test_predict_iris_versicolor(self):
        result =  self.model.predict(6.3, 3.3, 4.7, 1.6)
        print(result)
        self.assertEqual(result, "Iris-versicolor")

    def test_predict_iris_virginica(self):
        result =  self.model.predict(5.9, 3.0, 5.1, 1.8)
        print(result)
        self.assertEqual(result, "Iris-virginica")

if __name__ == '__main__':
    unittest.main()