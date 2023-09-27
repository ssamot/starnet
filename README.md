# starnet
Selected code for https://ieeexplore.ieee.org/abstract/document/9891910

You would need to plug it in the code into an evaluation framework -- see here for an example: https://github.com/misoc-mml/cate-benchmark (used in this paper https://arxiv.org/abs/2303.01412). 

The code is of "research quality", i.e. it is not adequately cleaned up and uses an older version of tf/keras.


```
from sklearn.ensemble import BaggingRegressor


model = CounterfactualNNRegressor(n_iterations=iterations, input_size=x_test.shape[1],
                                          build_model=build_model,
                                          batch_size=batch_size, plot=False, n_neurons=n_neurons, loss=loss,
                                          enable_propensity=propensity)

        
model = BaggingRegressor(base_estimator=model, n_estimators=n_estimators,
                                     bootstrap_features=False, verbose=0)


```