# run in Python shell
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.random.rand(100, 5)
y = np.random.randint(2, size=100)
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, "app/model.pkl")

