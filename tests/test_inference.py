"""Unit tests for model loading and inference."""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src import config
from src.architecture import MLPClassifier, MLPRegressor


class TestModelArchitecture:
    def test_regressor_output_shape(self):
        model = MLPRegressor(input_dim=37)
        x = torch.randn(8, 37)
        mu, log_sigma = model(x)
        assert mu.shape == (8,)
        assert log_sigma.shape == (8,)

    def test_classifier_output_shape(self):
        model = MLPClassifier(input_dim=37)
        x = torch.randn(8, 37)
        logits = model(x)
        assert logits.shape == (8,)

    def test_regressor_mu_range(self):
        """Mu should be unbounded (can represent any spread)."""
        model = MLPRegressor(input_dim=37)
        model.eval()
        x = torch.randn(100, 37)
        with torch.no_grad():
            mu, _ = model(x)
        # Should produce a range of values, not all the same
        assert mu.std() > 0

    def test_classifier_probabilities(self):
        """Sigmoid of logits should be in [0, 1]."""
        model = MLPClassifier(input_dim=37)
        model.eval()
        x = torch.randn(100, 37)
        with torch.no_grad():
            logits = model(x)
        probs = torch.sigmoid(logits)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0


class TestGaussianNLL:
    def test_loss_is_positive(self):
        from src.architecture import gaussian_nll_loss
        mu = torch.tensor([1.0, 2.0, 3.0])
        raw_sigma = torch.tensor([0.5, 0.5, 0.5])
        target = torch.tensor([1.5, 2.5, 3.5])
        loss = gaussian_nll_loss(mu, raw_sigma, target)
        assert loss.item() > 0

    def test_perfect_prediction_lower_loss(self):
        from src.architecture import gaussian_nll_loss
        mu_good = torch.tensor([1.0, 2.0, 3.0])
        mu_bad = torch.tensor([10.0, 20.0, 30.0])
        raw_sigma = torch.tensor([0.5, 0.5, 0.5])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss_good = gaussian_nll_loss(mu_good, raw_sigma, target)
        loss_bad = gaussian_nll_loss(mu_bad, raw_sigma, target)
        assert loss_good < loss_bad

    def test_sigma_clamping(self):
        """Very negative raw_sigma should still produce valid loss."""
        from src.architecture import gaussian_nll_loss
        mu = torch.tensor([1.0])
        raw_sigma = torch.tensor([-100.0])  # After softplus, almost 0 -> clamped to 0.5
        target = torch.tensor([1.0])
        loss = gaussian_nll_loss(mu, raw_sigma, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestCheckpointRoundtrip:
    def test_save_and_load_regressor(self, tmp_path):
        with patch.object(config, "CHECKPOINTS_DIR", tmp_path):
            from src.trainer import save_checkpoint

            model = MLPRegressor(input_dim=37, hidden1=64, hidden2=32)
            hp = {"hidden1": 64, "hidden2": 32, "dropout": 0.3}
            save_checkpoint(model, "test_reg", hparams=hp)

            ckpt = torch.load(tmp_path / "test_reg.pt", weights_only=False)
            assert ckpt["feature_order"] == config.FEATURE_ORDER
            assert ckpt["hparams"]["hidden1"] == 64

            loaded = MLPRegressor(input_dim=37, hidden1=64, hidden2=32)
            loaded.load_state_dict(ckpt["state_dict"])
            model.eval()
            loaded.eval()
            x = torch.randn(4, 37)
            with torch.no_grad():
                mu1, _ = model(x)
                mu2, _ = loaded(x)
            assert torch.allclose(mu1, mu2)

    def test_save_and_load_classifier(self, tmp_path):
        with patch.object(config, "CHECKPOINTS_DIR", tmp_path):
            from src.trainer import save_checkpoint

            model = MLPClassifier(input_dim=37, hidden1=64)
            save_checkpoint(model, "test_cls")

            ckpt = torch.load(tmp_path / "test_cls.pt", weights_only=False)
            loaded = MLPClassifier(input_dim=37, hidden1=64)
            loaded.load_state_dict(ckpt["state_dict"])
            loaded.eval()
            model.eval()
            x = torch.randn(4, 37)
            with torch.no_grad():
                out1 = model(x)
                out2 = loaded(x)
            assert torch.allclose(out1, out2)
