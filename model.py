import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# fake sample dataset (we will replace with real extracted features later)
X = [
    [0.1, 0.2, 0.3],
    [0.7, 0.8, 0.9],
    [0.4, 0.4, 0.5]
]

y = ["Healthy", "Rust", "Wilt"]

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("coffee_model.pkl", "wb"))

print("Model trained and saved!")