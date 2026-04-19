import joblib

from .base import BaseModel


class GenericSklearnModel(BaseModel):
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath: str):
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        self.model = joblib.load(filepath)
