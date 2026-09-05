"""NumPy inference for the validation-selected rank005 GNN checkpoint.

This intentionally implements only ``PhysicsInformedJetGraphClassifier`` in
evaluation mode.  It lets EOS Condor workers evaluate the frozen model without
shipping a multi-hundred-MB PyTorch runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _silu(values: np.ndarray) -> np.ndarray:
    return values / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _linear(values: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return values @ weight.T + bias


def _layer_norm(
    values: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    epsilon: float = 1.0e-5,
) -> np.ndarray:
    mean = np.mean(values, axis=-1, keepdims=True)
    variance = np.mean(np.square(values - mean), axis=-1, keepdims=True)
    return (values - mean) / np.sqrt(variance + epsilon) * weight + bias


class Rank005Numpy:
    def __init__(self, path: str | Path) -> None:
        payload = np.load(path)
        self.weights = {name: np.asarray(payload[name], dtype=np.float32) for name in payload.files}
        self.layers = int(np.asarray(self.weights.pop("__message_layers__")).item())
        self.hidden = int(np.asarray(self.weights.pop("__hidden__")).item())
        self.global_features = int(np.asarray(self.weights.pop("__global_features__")).item())

    def _encoder(self, nodes: np.ndarray) -> np.ndarray:
        w = self.weights
        hidden = _silu(_linear(nodes, w["encoder.0.weight"], w["encoder.0.bias"]))
        hidden = _linear(hidden, w["encoder.2.weight"], w["encoder.2.bias"])
        return _layer_norm(hidden, w["encoder.3.weight"], w["encoder.3.bias"])

    def _context(self, globals_: np.ndarray) -> np.ndarray:
        w = self.weights
        hidden = _silu(
            _linear(
                globals_,
                w["global_context.0.weight"],
                w["global_context.0.bias"],
            )
        )
        hidden = _linear(
            hidden,
            w["global_context.2.weight"],
            w["global_context.2.bias"],
        )
        return _layer_norm(
            hidden,
            w["global_context.3.weight"],
            w["global_context.3.bias"],
        )

    def _block(
        self,
        hidden: np.ndarray,
        eta: np.ndarray,
        phi: np.ndarray,
        mask: np.ndarray,
        index: int,
    ) -> np.ndarray:
        w = self.weights
        prefix = f"blocks.{index}."
        batch, nodes, width = hidden.shape
        hi = np.broadcast_to(hidden[:, :, None, :], (batch, nodes, nodes, width))
        hj = np.broadcast_to(hidden[:, None, :, :], (batch, nodes, nodes, width))
        deta = eta[:, :, None] - eta[:, None, :]
        raw_dphi = phi[:, :, None] - phi[:, None, :]
        sin_dphi = np.sin(raw_dphi)
        cos_dphi = np.cos(raw_dphi)
        wrapped = np.arctan2(sin_dphi, cos_dphi)
        delta_r = np.sqrt(np.square(deta) + np.square(wrapped) + 1.0e-8)
        geometry = np.stack((deta, sin_dphi, cos_dphi, delta_r), axis=-1)
        message_input = np.concatenate((hi, hj, geometry), axis=-1)
        messages = _silu(
            _linear(
                message_input,
                w[prefix + "message.0.weight"],
                w[prefix + "message.0.bias"],
            )
        )
        messages = _silu(
            _linear(
                messages,
                w[prefix + "message.2.weight"],
                w[prefix + "message.2.bias"],
            )
        )
        pair_mask = mask[:, :, None] & mask[:, None, :]
        pair_mask &= ~np.eye(nodes, dtype=bool)[None, :, :]
        messages = messages * pair_mask[..., None]
        count = np.maximum(pair_mask.sum(axis=2, keepdims=True), 1)
        mean = messages.sum(axis=2) / count
        maximum = np.max(np.where(pair_mask[..., None], messages, -1.0e9), axis=2)
        maximum = np.where(mask[:, :, None], maximum, 0.0)
        updated = _silu(
            _linear(
                np.concatenate((hidden, mean, maximum), axis=-1),
                w[prefix + "update.0.weight"],
                w[prefix + "update.0.bias"],
            )
        )
        updated = _linear(
            updated,
            w[prefix + "update.3.weight"],
            w[prefix + "update.3.bias"],
        )
        output = _layer_norm(
            hidden + updated,
            w[prefix + "norm.weight"],
            w[prefix + "norm.bias"],
        )
        return output * mask[..., None]

    def _predict_batch(
        self,
        nodes: np.ndarray,
        mask: np.ndarray,
        eta: np.ndarray,
        phi: np.ndarray,
        globals_: np.ndarray,
    ) -> np.ndarray:
        w = self.weights
        context = self._context(globals_)
        hidden = (self._encoder(nodes) + context[:, None, :]) * mask[..., None]
        for index in range(self.layers):
            hidden = self._block(hidden, eta, phi, mask, index)
        count = np.maximum(mask.sum(axis=1, keepdims=True), 1)
        mean = hidden.sum(axis=1) / count
        maximum = np.max(np.where(mask[..., None], hidden, -1.0e9), axis=1)
        maximum = np.where(np.isfinite(maximum), maximum, 0.0)
        head = _silu(
            _linear(
                np.concatenate((mean, maximum, context, globals_), axis=-1),
                w["head.0.weight"],
                w["head.0.bias"],
            )
        )
        head = _silu(_linear(head, w["head.3.weight"], w["head.3.bias"]))
        logits = _linear(head, w["head.6.weight"], w["head.6.bias"]).reshape(-1)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))

    def predict(
        self,
        nodes: np.ndarray,
        mask: np.ndarray,
        eta: np.ndarray,
        phi: np.ndarray,
        globals_: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        if globals_.shape[1] != self.global_features:
            raise ValueError("global-feature width does not match frozen model")
        output = np.empty(len(nodes), dtype=np.float32)
        for start in range(0, len(nodes), batch_size):
            stop = min(start + batch_size, len(nodes))
            output[start:stop] = self._predict_batch(
                np.asarray(nodes[start:stop], dtype=np.float32),
                np.asarray(mask[start:stop], dtype=bool),
                np.asarray(eta[start:stop], dtype=np.float32),
                np.asarray(phi[start:stop], dtype=np.float32),
                np.asarray(globals_[start:stop], dtype=np.float32),
            )
        return output
