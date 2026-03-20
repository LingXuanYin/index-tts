import torch

from indextts.s2mel.modules.flow_matching import BASECFM


class _DummyArgs:
    reg_loss_type = "l2"

    class DiT:
        in_channels = 80


class _DummyCFM(BASECFM):
    def __init__(self):
        super().__init__(_DummyArgs())
        self.estimator = lambda x, prompt_x, x_lens, t, style, mu: torch.zeros_like(x)


def test_solve_euler_clamps_prompt_to_target_length():
    cfm = _DummyCFM()
    x = torch.randn(1, 80, 5)
    x_lens = torch.tensor([5])
    prompt = torch.randn(1, 80, 8)
    mu = torch.randn(1, 5, 4)
    style = torch.randn(1, 192)
    t_span = torch.linspace(0, 1, 2)

    out = cfm.solve_euler(x, x_lens, prompt, mu, style, None, t_span)

    assert out.shape == x.shape
