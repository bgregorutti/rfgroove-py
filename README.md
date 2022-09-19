# RFgroove - Importance Measure and Selection for Groups of Variables with Random Forests

Implement an importance measure for groups of features and the Recursive Feature Elimination algorithm (from Guyon et al. 2002).

Based on the two following articles:
* B. Gregorutti, B. Michel, P. Saint-Pierre (2017). Correlation and variable importance in random forests. [arXiv link](https://arxiv.org/abs/1310.5726)
* B. Gregorutti, B. Michel, P. Saint-Pierre (2015). Grouped variable importance with random forests and application to multiple functional data analysis. [arXiv link](https://arxiv.org/abs/1411.4170)
* I. Guyon, J. Weston, S. Barnhill, & V. Vapnik (2002). Gene selection for cancer classification using support vector machines, Mach. Learn., 46(1-3), 389–422.


### REQUIREMENTS

* numpy
* joblib
* scikit-learn


### INSTALLATION

```bash
git clone git@github.com:bgregorutti/rfgroove-py.git
cd rfgroove-py/
pip install .
```


### CODE EXAMPLES

***Feature importance:***

```python
from sklearn.ensemble import RandomForestRegressor
from rfgroove.dataset_generation import gaussian_multidimensional
from rfgroove.importance import grouped_importance

# Build a dataset
X, y = gaussian_multidimensional(c=.5, size=1000, n_features_per_groups=5)

# Fit a RF model
regr = RandomForestRegressor(n_estimators=100, oob_score=True, max_samples=.1)
regr.fit(X, y)

# Compute the feature importance measure
groups = [list(range(5)), list(range(5, 10))] + [[k] for k in range(10, 15)]
imp = grouped_importance(regr, X, y, groups)
```

see `test/test_selection.py`.



***Feature selection:***

```python
from sklearn.ensemble import RandomForestRegressor
from rfgroove.dataset_generation import gaussian_multidimensional
from rfgroove.selection import RFE
    
# Build a dataset
X, y = gaussian_multidimensional(c=.5, size=1000, n_features_per_groups=5)

# Instanciate a RandomForestRegressor object, as base model
base = RandomForestRegressor(n_estimators=1000, bootstrap=True, oob_score=True, max_samples=.1)

# Run the selection algorithm
groups = [list(range(5)), list(range(5, 10))] + [[k] for k in range(10, 15)]
selector = RFE(base, groups, n_jobs=-1)
selector.fit(X, y)
```

see `test/test_importance.py`.