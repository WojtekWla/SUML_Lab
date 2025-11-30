import streamlit as st

from predict import IrisModel

class IrisWebsite:
    def __init__(self):
        self.model = IrisModel('model.joblib')
        self.image_map = {
            "Iris-setosa": "assets/is.png",
            "Iris-versicolor": "assets/iver.png",
            "Iris-virginica": "assets/ivir.png"
        }

    @st.dialog("Error negative or 0 value")
    def error_dialog(self):
        st.write("One of provided values is not positive")

    @st.dialog("Your Iris")
    def predict_and_display(self, sl, sw, pl, pw):
        output_c = st.container(horizontal_alignment="center")
        prediction = self.model.predict(sl, sw, pl, pw)
        output_c.image(self.image_map[prediction], caption=prediction )

    def main(self):
        st.title("Predict you Iris")
        container = st.container(border=True)
        sepal_length = container.number_input("Pepal length")
        sepal_width = container.number_input("Sepal width")
        petal_length = container.number_input("Petal length")
        petal_width = container.number_input("Petal width")

        button_container = container.container(horizontal_alignment="center")

        if button_container.button("Predict"):
            if sepal_length <= 0 or sepal_width <= 0 or petal_length <= 0 or petal_width <= 0:
                self.error_dialog()
            else:
                self.predict_and_display(sepal_length, sepal_width, petal_length, petal_width)






website = IrisWebsite()
website.main()