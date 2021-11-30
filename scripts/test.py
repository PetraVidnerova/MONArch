import numpy as np
import mobopt as mo


def objective(x):
    return np.array(
        [
            -x[0]*x[0],
            -abs(x[0])
        ]
    )


optimizer = mo.MOBayesianOpt(target=objective,
                             NObj=2,
                             pbounds=np.array(
                                 [-10, 10], dtype=float).reshape(1, 2),
                             verbose=True,
                             max_or_min='max'
                             )

optimizer.initialize(init_points=2)

front, pop = optimizer.maximize(n_iter=50)

print(pop)
