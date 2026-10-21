import inspect
import numpy as np
import torch
from src.models import SmallLSTM, train_models, predict_models

def test_lstm_independent_windows_and_gradient():
    torch.manual_seed(2026)
    m=SmallLSTM()
    x=torch.randn(3,24)
    a=m(x)
    assert a.shape==(3,)
    assert torch.allclose(a[:1],m(x[:1]),atol=1e-6)
    loss=((a-torch.tensor([0.,1.,2.]))**2).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_training_has_no_test_labels_interface():
    assert list(inspect.signature(train_models).parameters)==['train_x','train_y','val_x','val_y','val_regions','transform','cfg','model_dir']
    assert list(inspect.signature(predict_models).parameters)==['bundle','x','transform']
