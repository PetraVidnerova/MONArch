import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping    
from sklearn.model_selection import KFold


def _accuracy(y_true, y_hat):
        y_true = y_true.argmax(axis=1)
        y_hat = y_hat.argmax(axis=1)
        return np.mean(y_true == y_hat)

def crossval_objective(space, D, x, stopping=True, patience=3,  epochs=10):

    X, y = D

    kf = KFold(n_splits=3, random_state=42, shuffle=True)
    scores = []
    for train, test in kf.split(X):   # train, test are indicies
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        net = space.create_network(x.body)
        if net is None:
                return 0, -100000
            
        net.compile(
            loss="categorical_crossentropy",
            optimizer=Adam()
        )

        if stopping:
            callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]
        else:
            callbacks = []
        
        net.fit(X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=128,
                callbacks=callbacks
        )
        yhat = net.predict(X_test)
        scores.append(_accuracy(y_test, yhat))

    ret = [100*sum(scores)/len(scores), -(net.count_params() // 1000)]
    print("OBJECTIVE:", ret)

    return ret
                    

def objective(space, D, x, stopping=True, epochs=10, patience=3):

    net = space.create_network(x)

    net.compile(
        loss="categorical_crossentropy",
        optimizer=Adam()
    )

    if stopping:
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]
    else:
        callbacks = []
        
    X_train, y_train, X_test, y_test = D
    net.fit(X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=128,
            callbacks=callbacks
    )
    yhat = net.predict(X_test)

    ret = [_accuracy(y_test, yhat), -(net.count_params())]
    print("OBJECTIVE:", ret, x)

    return ret[0]
