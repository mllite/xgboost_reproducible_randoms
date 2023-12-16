import numpy as np
import pandas as pd
import sys

# ******************************************************
from sklearn.datasets import load_iris
import xgboost as xgb

iris = load_iris()
X = iris.data
y = iris.target

clf = xgb.XGBClassifier(n_estimators=1, nthread=1, min_child_weight=10, max_depth=3, objective="multi:softmax", num_class=3, seed=1789);

clf.max_bin = 10
lDict = clf.__dict__

print("XGBOOST_CLASS_OPTIONS_START")
for x in sorted(lDict.items()):
    print("XGBOOST_CLASS_OPTION" , x)
print("XGBOOST_CLASS_OPTIONS", [x[0] for x in sorted(lDict.items())])
print("XGBOOST_CLASS_OPTIONS_END")

sys.stdout.flush();

clf.fit(X, y)

print(clf.__dict__)

lJSon = clf.get_booster().get_dump(with_stats=True, dump_format='json')

for x in lJSon:
    print(x)

pred = clf.predict(X)
print(pred)

proba = clf.predict_proba(X)
print(proba)
