import numpy as np
from src.pipeline import create_pipeline

def test_pipeline_fit_predict():
    # tiny synthetic dataset with 6 rows and 4 features
    X = np.array([
        [1.0, 0.1, 0.2, 0.3],
        [0.9, 0.2, 0.1, 0.4],
        [1.2, 0.1, 0.2, 0.5],
        [0.2, 0.4, 0.6, 0.7],
        [0.3, 0.5, 0.8, 0.2],
        [0.4, 0.6, 0.9, 0.1],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    pipe = create_pipeline(k=2)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == X.shape[0]
