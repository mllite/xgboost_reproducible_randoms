
import numpy as np
import sys

from sklearn import datasets
import xgboost as xgb

X, y = datasets.load_diabetes(return_X_y=True)

for i in range(X.shape[0]):
    if(i < 12):
        print("DIABETES_DATA", i, X[i, :].tolist(), y[i])

print("y_mean = ", y.mean())
clf = xgb.XGBRegressor(n_estimators=1, nthread=1, min_child_weight=10, max_depth=6, seed=1960);

clf.max_bin = 10
lDict = clf.__dict__

print("XGBOOST_REG_OPTIONS_START")
for x in sorted(lDict.items()):
    print("XGBOOST_REG_OPTION" , x)
print("XGBOOST_REG_OPTIONS", [x[0] for x in sorted(lDict.items())])
print("XGBOOST_REG_OPTIONS_END")

sys.stdout.flush();

clf.fit(X, y)

print(clf.__dict__)

lJSon = clf.get_booster().get_dump(with_stats=True, dump_format='json')

for x in lJSon:
    print(x)

pred = clf.predict(X)
print(pred)
