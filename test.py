from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

X,y = datasets.load_iris(return_X_y=True, as_frame=True)

print("Info: ", X.info())
print("Describe: ", X.describe())
print("Head: ", X.head())
print("Correlation", X.corr())

print("Value counts:", y.value_counts())



# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# model = keras.Sequential([
#     layers.Dense(1000, activation='relu', input_shape=(X_train.shape[1],)),
#     layers.Dense(500, activation='relu'),
#     layers.Dense(300, activation='relu'),
#     layers.Dropout(0.1),
#     layers.Dense(3, activation='softmax')
# ])
#
# model.summary()
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(X_train, Y_train, epochs=30, validation_data=(X_test, Y_test))
# model.evaluate(X_test, Y_test)
#
# pd.DataFrame(history.history).plot(figsize=(8, 5))
# plt.grid(True)
# plt.savefig("logs/history.png")
#
# joblib.dump(model, "model.joblib")
