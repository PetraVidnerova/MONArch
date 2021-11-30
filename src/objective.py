from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import RMSprop


def objective(encoding, D, code):

    net = encoding.decode(code)

    net.compile(
        loss="categorical_crossentropy",
        optimizer=RMSprop()
    )

    X_train, y_train, X_test, y_test = D
    net.fit(X_train, y_train, epochs=10)
    yhat = net.predict(X_test)

    metrics = CategoricalAccuracy()
    parameters = net.count_params()//10000

    return -metrics(y_test, yhat), -parameters
