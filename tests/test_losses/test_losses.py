import pytest
import torch
from traiNNer.losses.basic_loss import *


class TestLosses:
    N = 1
    H = 256
    W = 256

    black_image = torch.tensor([[[0, 0, 0]]], dtype=torch.float32).repeat(N, H, W, 1).permute(0, 3, 1, 2)
    white_image = torch.tensor([[[1, 1, 1]]], dtype=torch.float32).repeat(N, H, W, 1).permute(0, 3, 1, 2)
    red_image = torch.tensor([[[1, 0, 0]]], dtype=torch.float32).repeat(N, H, W, 1).permute(0, 3, 1, 2)
    green_image = torch.tensor([[[0, 1, 0]]], dtype=torch.float32).repeat(N, H, W, 1).permute(0, 3, 1, 2)

    mssim_neo_loss = MSSIMNeoLoss()
    l1_loss = L1Loss()
    luma_loss = LumaLoss(criterion="charbonnier")
    charbonnier_loss = CharbonnierLoss()
    perceptual_loss = PerceptualLoss(
        layer_weights={"conv1_2": 0.1, "conv2_2": 0.1, "conv3_4": 1, "conv4_4": 1, "conv5_4": 1})
    color_loss = ColorLoss(criterion="charbonnier")
    dists_loss = DISTSLoss()

    eps = 1e-5  # torch.finfo(torch.float32).eps

    @pytest.mark.parametrize('loss_class', [L1Loss, MSELoss, CharbonnierLoss])
    def test_pixellosses(self, loss_class):
        """Test loss: pixel losses"""

        pred = torch.rand((1, 3, 4, 4), dtype=torch.float32)
        target = torch.rand((1, 3, 4, 4), dtype=torch.float32)
        loss = loss_class(loss_weight=1.0, reduction='mean')
        out = loss(pred, target, weight=None)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([])

        # -------------------- test with other reduction -------------------- #
        # reduction = none
        loss = loss_class(loss_weight=1.0, reduction='none')
        out = loss(pred, target, weight=None)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 3, 4, 4)
        # test with spatial weights
        weight = torch.rand((1, 3, 4, 4), dtype=torch.float32)
        out = loss(pred, target, weight=weight)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 3, 4, 4)

        # reduction = sum
        loss = loss_class(loss_weight=1.0, reduction='sum')
        out = loss(pred, target, weight=None)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([])

        # -------------------- test unsupported loss reduction -------------------- #
        with pytest.raises(ValueError):
            loss_class(loss_weight=1.0, reduction='unknown')

    def test_weightedtvloss(self):
        """Test loss: WeightedTVLoss"""

        pred = torch.rand((1, 3, 4, 4), dtype=torch.float32)
        loss = WeightedTVLoss(loss_weight=1.0, reduction='mean')
        out = loss(pred, weight=None)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([])

        # test with spatial weights
        weight = torch.rand((1, 3, 4, 4), dtype=torch.float32)
        out = loss(pred, weight=weight)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([])

        # -------------------- test reduction = sum-------------------- #
        loss = WeightedTVLoss(loss_weight=1.0, reduction='sum')
        out = loss(pred, weight=None)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([])

        # test with spatial weights
        weight = torch.rand((1, 3, 4, 4), dtype=torch.float32)
        out = loss(pred, weight=weight)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([])

        # -------------------- test unsupported loss reduction -------------------- #
        with pytest.raises(ValueError):
            WeightedTVLoss(loss_weight=1.0, reduction='unknown')
        with pytest.raises(ValueError):
            WeightedTVLoss(loss_weight=1.0, reduction='none')

    @pytest.mark.parametrize("loss_fn",
                             [mssim_neo_loss, l1_loss, luma_loss, mse_loss, charbonnier_loss, perceptual_loss,
                              color_loss, dists_loss])
    def test_black_vs_black(self, loss_fn):
        loss_value = loss_fn(self.black_image, self.black_image)
        print(loss_value)

        if type(loss_value) is tuple:
            assert loss_value[0] <= self.eps
        else:
            assert loss_value <= self.eps

    @pytest.mark.parametrize("loss_fn",
                             [mssim_neo_loss, l1_loss, luma_loss, mse_loss, charbonnier_loss, perceptual_loss,
                              color_loss, dists_loss])
    def test_black_vs_black_float64(self, loss_fn):
        loss_value = loss_fn(self.black_image.to(dtype=torch.float64), self.black_image.to(dtype=torch.float64))
        print(loss_value)

        if type(loss_value) is tuple:
            assert loss_value[0] <= self.eps
        else:
            assert loss_value <= self.eps
