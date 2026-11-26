"""Loss functions for SHERLOC and the deep competitors."""
import torch


class SafeCoxPHLoss(torch.nn.Module):
    """Negative Cox partial log-likelihood that is defined on every mini-batch.

    ``pycox.models.loss.CoxPHLoss`` divides by the number of events in the mini-batch. A batch
    without any event (for example a small remainder batch) gives 0/0 = NaN; the weights then
    become NaN and ``EarlyStopping`` silently restores the last finite epoch, so the network
    barely trains. Here an event-free batch contributes 0 (with a zero gradient).

    Usage::

        model = CoxPH(net, tt.optim.Adam, loss=SafeCoxPHLoss())
    """

    def forward(self, log_h: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        events = events.float().view(-1)
        if events.sum() == 0:
            return log_h.sum() * 0.0
        from pycox.models.loss import cox_ph_loss
        return cox_ph_loss(log_h, durations, events)
