"""Reproduce https://github.com/pytorch/pytorch/issues/130295

argmax/argmin ignore NaN on MPS. CPU returns the index of the first NaN
(NaN propagates through max/min).
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestArgmaxArgminNaN(TestCase):
    def test_argmax_with_nan(self):
        x = torch.tensor([1.0, float("nan"), 3.0], device="mps")
        x_cpu = x.cpu()
        self.assertEqual(torch.argmax(x).item(), torch.argmax(x_cpu).item())
        self.assertEqual(torch.argmin(x).item(), torch.argmin(x_cpu).item())

    def test_argmax_dim_with_nan(self):
        x = torch.tensor(
            [[1.0, float("nan"), 3.0], [float("nan"), 2.0, 1.0], [4.0, 5.0, 6.0]],
            device="mps",
        )
        x_cpu = x.cpu()
        self.assertEqual(torch.argmax(x, dim=0).cpu(), torch.argmax(x_cpu, dim=0))
        self.assertEqual(torch.argmax(x, dim=1).cpu(), torch.argmax(x_cpu, dim=1))
        self.assertEqual(torch.argmin(x, dim=0).cpu(), torch.argmin(x_cpu, dim=0))
        self.assertEqual(torch.argmin(x, dim=1).cpu(), torch.argmin(x_cpu, dim=1))

    def test_argmax_no_nan(self):
        x = torch.randn(100, device="mps")
        x_cpu = x.cpu()
        self.assertEqual(torch.argmax(x).item(), torch.argmax(x_cpu).item())
        self.assertEqual(torch.argmin(x).item(), torch.argmin(x_cpu).item())

    def test_argmax_all_nan(self):
        x = torch.tensor([float("nan"), float("nan"), float("nan")], device="mps")
        x_cpu = x.cpu()
        self.assertEqual(torch.argmax(x).item(), torch.argmax(x_cpu).item())

    def test_argmax_half_nan(self):
        # float16 also affected
        x = torch.tensor([1.0, float("nan"), 3.0], device="mps", dtype=torch.float16)
        x_cpu = x.cpu()
        self.assertEqual(torch.argmax(x).item(), torch.argmax(x_cpu).item())


if __name__ == "__main__":
    run_tests()
