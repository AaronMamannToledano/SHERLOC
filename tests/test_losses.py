"""SafeCoxPHLoss: finite on event-free mini-batches, identical to pycox otherwise."""
import torch
import pytest

pycox = pytest.importorskip("pycox")
from pycox.models.loss import cox_ph_loss  # noqa: E402

from sherloc.losses import SafeCoxPHLoss  # noqa: E402


def test_event_free_batch_is_finite_and_zero():
    log_h = torch.randn(8, 1, requires_grad=True)
    durations = torch.rand(8)
    events = torch.zeros(8)
    loss = SafeCoxPHLoss()(log_h, durations, events)
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()
    assert torch.all(log_h.grad == 0)
    # the default pycox loss is NaN on the same batch
    assert torch.isnan(cox_ph_loss(log_h.detach(), durations, events))


def test_same_as_pycox_with_events():
    torch.manual_seed(0)
    log_h = torch.randn(16, 1)
    durations = torch.rand(16)
    events = (torch.rand(16) > 0.5).float()
    events[0] = 1
    assert torch.allclose(SafeCoxPHLoss()(log_h, durations, events), cox_ph_loss(log_h, durations, events))
