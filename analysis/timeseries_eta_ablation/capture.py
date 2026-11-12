"""Forward hooks for Primal.+Trans. No repo source is modified."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = os.path.join(REPO, "PrimalAttention_TimeSeries", "Primal.+Trans.")
if CODE not in sys.path:
    sys.path.insert(0, CODE)

import torch  # noqa: E402


class Capture:
    """Records one forward pass. Use as a context manager so hooks are always removed.

    Keys after a forward pass:
      x      (B, N, d_model)  tokens entering layer 1
      phi_q, phi_k (B, H, N, p)  normalized feature maps
      E, R   (B, H, N, s)
      We, Wr (H, n, s)
      lam    (H, s)
      attn0  (B, H, N, N)  post-softmax matrix of layer 0, if softmax=True
    """

    def __init__(self, model, softmax=True):
        self.d = {}
        self.layer1 = model.encoder.attn_layers[1].attention
        self.handles = [
            self.layer1.register_forward_pre_hook(self._tokens),
            self.layer1.inner_attention.register_forward_hook(self._primal),
        ]
        if softmax:
            # SoftmaxAttention keeps attn in a local variable; its nn.Dropout receives it.
            dropout = model.encoder.attn_layers[0].attention.inner_attention.dropout
            self.handles.append(dropout.register_forward_hook(self._softmax))

    def _softmax(self, _module, inputs, _output):
        self.d["attn0"] = inputs[0].detach()

    def _tokens(self, _module, inputs):
        self.d["x"] = inputs[0].detach()

    def _primal(self, _module, _inputs, output):
        _, s, lam = output
        self.d.update(E=s[0].detach(), R=s[1].detach(), We=s[2].detach(), Wr=s[3].detach(),
                      phi_q=s[4].detach(), phi_k=s[5].detach(), lam=lam.detach())

    def fx(self):
        """F_X as AttentionLayer.forward builds it (Primal_Attention.py:107-109): (B, H, n, p)."""
        x = self.d["x"]
        B, L, _ = x.shape
        H, p, n = self.layer1.n_heads, self.layer1.d_keys, self.layer1.len
        idx = torch.linspace(0, L - 1, n, dtype=int)
        return x.transpose(-2, -1).reshape(B, H, p, L)[:, :, :, idx].transpose(-2, -1)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
