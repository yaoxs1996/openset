import gan
import data_loader
import numpy as np
from tensorflow.keras import losses, optimizers

def data_process():
    x_train, y_train, x_test, y_test = data_loader.load_data()
    y_train = y_train.astype(np.int).copy()
    y_test = y_test.astype(np.int).copy()

    y_train[y_train <= 6] = 1
    y_train[y_train > 6] = 0

    y_test[y_test <= 6] = 1
    y_test[y_test > 6] = 0

    return x_train, y_train, x_test, y_test

def model():
    x_train, y_train, x_test, y_test = data_process()
    discminator = gan.make_discriminator(x_train)
    discminator.compile(optimizer=optimizers.Adam(1e-4), loss=losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
    discminator.fit(x_train, y_train, epochs=10)
    discminator.evaluate(x_test, y_test)

if __name__ == "__main__":
    model()