import math

import torch
from torch import Tensor, autograd, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils.registry import LOSS_REGISTRY

USE_BOOL_TARGET = {"wgan", "wgan_softplus"}


@LOSS_REGISTRY.register()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(
        self,
        loss_weight: float,
        gan_type: str = "vanilla",
        real_label_val: float = 1.0,
        fake_label_val: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.eps = eps

        if self.gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "wgan":
            self.loss = self._wgan_loss
        elif self.gan_type == "wgan_softplus":
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == "hinge":
            self.loss = nn.ReLU()
        elif self.gan_type == "ganetic":
            self.loss = self._ganetic_loss
        else:
            raise NotImplementedError(f"GAN type {self.gan_type} is not implemented.")

    def _wgan_loss(self, input: Tensor, target: bool | Tensor) -> Tensor:
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        assert isinstance(target, bool)
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input: Tensor, target: bool | Tensor) -> Tensor:
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        assert isinstance(target, bool)
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def _ganetic_loss(self, x: Tensor, target: Tensor | bool) -> Tensor:
        assert isinstance(target, Tensor)
        if x.shape != target.shape:
            raise ValueError(
                "Input and target must have the same shape for GANetic loss"
            )

        x = torch.sigmoid(x)
        loss = x**3 + torch.sqrt(
            torch.abs(3.985 * target / (torch.sum(x) + self.eps)) + self.eps
        )
        return loss.mean()

    def get_target_label(self, input: Tensor, target_is_real: bool) -> Tensor | bool:
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in USE_BOOL_TARGET:
            return target_is_real
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_ones(input.size()) * target_val

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(
        self, input: Tensor, target_is_real: bool, is_disc: bool = False
    ) -> Tensor:
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        if self.gan_type == "hinge":
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                assert isinstance(self.loss, nn.ReLU)
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            target_label = self.get_target_label(input, target_is_real)
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


@LOSS_REGISTRY.register()
class MultiScaleGANLoss(GANLoss):
    """
    MultiScaleGANLoss accepts a list of predictions
    """

    def __init__(
        self,
        loss_weight: float,
        gan_type: str,
        real_label_val: float = 1.0,
        fake_label_val: float = 0.0,
    ) -> None:
        super().__init__(loss_weight, gan_type, real_label_val, fake_label_val)

    def forward(
        self, input: Tensor | list[Tensor], target_is_real: bool, is_disc: bool = False
    ) -> Tensor:
        """
        The input is a list of tensors, or a list of (a list of tensors)
        """
        if isinstance(input, list):
            assert len(input) > 0
            loss = torch.tensor(0, device=input[0].device)
            for pred_i_wrapper in input:
                pred_i = pred_i_wrapper
                if isinstance(pred_i, list):
                    # Only compute GAN loss for the last layer
                    # in case of multiscale feature matching
                    pred_i = pred_i[-1]
                # Safe operation: 0-dim tensor calling self.mean() does nothing
                loss_tensor = super().forward(pred_i, target_is_real, is_disc).mean()
                loss += loss_tensor
            return loss / len(input)
        else:
            return super().forward(input, target_is_real, is_disc)


def r1_penalty(real_pred: Tensor, real_img: Tensor) -> Tensor:
    """R1 regularization for discriminator. The core idea is to
    penalize the gradient on real data alone: when the
    generator distribution produces the true data distribution
    and the discriminator is equal to 0 on the data manifold, the
    gradient penalty ensures that the discriminator cannot create
    a non-zero gradient orthogonal to the data manifold without
    suffering a loss in the GAN game.

    Reference: Eq. 9 in Which training methods for GANs do actually converge.
    """
    grad_real = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(
    fake_img: Tensor, latents: Tensor, mean_path_length: Tensor, decay: float = 0.01
) -> tuple[Tensor, Tensor, Tensor]:
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(
    discriminator: nn.Module,
    real_data: Tensor,
    fake_data: Tensor,
    weight: Tensor | None = None,
) -> Tensor:
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1.0 - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty
