import numpy as np
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

def _accuracy(y_true, y_hat):

    y_true = y_true.argmax(axis=1)
    y_hat = y_hat.argmax(axis=1)

    return np.mean(y_true == y_hat)

def objective(encoding, D, code, stopping=True, epochs=100):

    net = encoding.decode(code)

    net.compile(
        loss="categorical_crossentropy",
        optimizer=RMSprop()
    )

    if stopping:
        callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
    else:
        callbacks = []
        
    X_train, y_train, X_test, y_test = D
    net.fit(X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=512,
            callbacks=callbacks
    )
    yhat = net.predict(X_test)

    ret = [_accuracy(y_test, yhat), -(net.count_params()//1000)]
    print("*** OBJECTIVE *** :", ret)

    return ret
