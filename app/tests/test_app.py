import app.predict as p
import unittest

class Tests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Tests, self).__init__(*args, **kwargs)
        self.model = p.IrisModel('./app/model.joblib')

    def test_predict_iris_setosa(self):
        result = self.model.predict(5.1, 3.5, 1.4, 0.2)
        result_2 = self.model.predict(4.9, 3.0, 1.4, 0.2)
        self.assertEqual(result, "Iris-setosa")
        self.assertEqual(result_2, "Iris-setosa")



    def test_predict_iris_versicolor(self):
        result =  self.model.predict(6.3, 3.3, 4.7, 1.6)
        result_2 = self.model.predict(6.7, 3.1, 4.4, 1.4)
        self.assertEqual(result, "Iris-versicolor")
        self.assertEqual(result_2, "Iris-versicolor")

    def test_predict_iris_virginica(self):
        result =  self.model.predict(5.9, 3.0, 5.1, 1.8)
        result_2 =  self.model.predict(7.7, 2.8, 6.7, 2.0)
        self.assertEqual(result, "Iris-virginica")
        self.assertEqual(result_2, "Iris-virginica")

if __name__ == '__main__':
    unittest.main()