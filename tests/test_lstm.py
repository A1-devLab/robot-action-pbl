import numpy as np
import pytest

pytest.importorskip("torch")

from src.lstm import LSTMClassifier, augment_sequences  # noqa: E402


@pytest.mark.parametrize("pooling", ["last", "mean"])
def test_learns_a_simple_sequence_rule(pooling):
    # label = whether channel 0 rises at the end of the sequence
    rng = np.random.default_rng(1)
    y = np.tile([0, 1], 20)
    X = rng.normal(0, 0.3, size=(40, 20, 2))
    X[:, 15:, 0] += np.where(y == 1, 1.5, -1.5)[:, None]
    model = LSTMClassifier(hidden=8, epochs=120, dropout=0.0, pooling=pooling).fit(X[:30], y[:30])
    assert (model.predict(X[30:]) == y[30:]).mean() >= 0.9
    assert model.loss_curve_[-1] < model.loss_curve_[0]


def test_same_seed_gives_same_predictions():
    rng = np.random.default_rng(4)
    X, y = rng.normal(size=(20, 10, 3)), np.tile([0, 1], 10)
    a = LSTMClassifier(hidden=4, epochs=20, n_augment=2).fit(X, y).predict_proba(X)
    b = LSTMClassifier(hidden=4, epochs=20, n_augment=2).fit(X, y).predict_proba(X)
    assert np.allclose(a, b)


def test_augmentation_keeps_shape_and_labels():
    X, y = np.random.default_rng(2).normal(size=(4, 10, 3)), np.array([0, 1, 0, 1])
    Xa, ya = augment_sequences(X, y, n_copies=3)
    assert Xa.shape == (16, 10, 3) and ya.tolist() == y.tolist() * 4
    assert np.allclose(Xa[:4], X)


def test_early_stopping_keeps_best_validation_epoch():
    rng = np.random.default_rng(3)
    y = np.tile([0, 1], 15)
    X = rng.normal(size=(30, 10, 2))
    model = LSTMClassifier(hidden=4, epochs=40, val_fraction=0.2).fit(X, y)
    assert len(model.val_curve_) == 40
    assert model.val_curve_[model.best_epoch_ - 1] == min(model.val_curve_)
