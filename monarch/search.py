import numpy as np 
from scipy.stats import norm

from surrogate import SurrogateModel 

class NeMOBOptimizer():

    def __init__(self,
                 objective,
                 space
    ):
        
        self.objective = objective
        self.generator = space.get_random_sample
        self.get_features = space.get_features
        self.combinator = space.select_and_crossover
        
    def aquisition(self, individuals, X, candidates, codes):
        yhat = self.model.predict(X)
        best = np.max(yhat)

        mu, std = self.model.predict(codes, return_std=True)
        mu = mu[:, 0]
        probs = norm.cdf((mu - best) / (std+1E-9))
        return probs

    def opt_aquisition(self, individuals, X, y):

        candidates = self.combinator(individuals, y, 30)
        # candidates.extend([
        #     self.generator()
        #     for _ in range(10)
        # ])
        codes = np.vstack([ self.get_features(x) for x in candidates ])
        
        scores = self.aquisition(individuals, X, candidates, codes) 
        couples = list(zip(candidates, scores))
        couples.sort(reverse=True, key=lambda x: x[1])
        return couples[:1]
        
        # idx = np.argmax(scores)
        # print(scores)
        # print("scores", scores.shape)
        # print("len scores", len(scores))
        # print("len candidates", len(candidates))
        # print("lend codes", len(codes))
        # print(idx)
        # return candidates[idx], codes[idx]
        
    def optimize(self,
                 inital_points_num=6,
                 n_steps=100):
    
        individuals = [ self.generator() for _ in range(inital_points_num) ]
        y = [ self.objective(x) for x in individuals ]
        X = [ self.get_features(x) for x in individuals] 

        y = np.array(y).reshape(-1,1)
        X = np.vstack(X)
        
        self.model = SurrogateModel()
        self.model.fit(X, y)
        
        for i in range(n_steps):

            print(f" *** Optimization step {i} ***, best so far {np.max(y)}")

            couples = self.opt_aquisition(individuals, X, y)
            new_inds = [ x for x, _ in couples ]
            new_codes = np.vstack([ self.get_features(x) for x in new_inds ])
            
            objs = []
            for ind in new_inds:
                obj = self.objective(ind)
                objs.append(obj)

            est = self.model.predict(new_codes)
            print(f"estimated: {est[0]}, real: {objs[0]}")
            print(f"est error: {abs(obj-est)}")

            individuals.extend(new_inds)
            X = np.vstack((X, new_codes))
            objs = np.array(objs).reshape(-1,1)
            y = np.vstack((y, objs))

            self.model.fit(X, y)
        
        idx = np.argmax(y.ravel())
        return individuals[idx]
