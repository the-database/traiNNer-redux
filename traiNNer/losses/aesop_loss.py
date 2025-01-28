from typing import Literal

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from traiNNer.archs.autoencoder_arch import AutoEncoder
from traiNNer.losses.basic_loss import L1Loss, charbonnier_loss
from traiNNer.losses.ms_ssim_l1_loss import MSSSIML1Loss
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class AESOPLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float,
        scale: int,
        pretrain_network_ae: str,
        criterion: Literal["l1", "charbonnier", "msssiml1"] = "msssiml1",
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.ae = AutoEncoder(freeze=True, scale=scale)
        self.ae.load_state_dict(
            load_file(pretrain_network_ae)
        )  # TODO wrapper function to support pth/safetensors
        if criterion == "l1":
            self.criterion = L1Loss(1.0)
        elif criterion == "charbonnier":
            self.criterion = charbonnier_loss(1.0)
        else:
            self.criterion = MSSSIML1Loss(1.0)

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        # ae_sr = self.ae(sr.clamp(0, 1)).clamp(0, 1)
        # ae_hr = self.ae(hr.detach()).clamp(0, 1)
        ae_sr = self.ae(sr)
        ae_hr = self.ae(hr.detach())
        return self.loss_weight * self.criterion(ae_sr, ae_hr)


# debugging
# @LOSS_REGISTRY.register()
# class AESOPLoss(nn.Module):
#     def __init__(
#         self,
#         loss_weight: float,
#         scale: int,
#         pretrain_network_ae: str,
#         criterion: Literal["l1", "charbonnier", "msssiml1"] = "msssiml1",
#     ) -> None:
#         super().__init__()
#         self.loss_weight = loss_weight
#         self.debug_dir = r""  # TODO

#         self.ae = AutoEncoder(freeze=True, scale=scale)
#         self.ae.load_state_dict(
#             load_file(pretrain_network_ae)
#         )  # TODO wrapper function to support pth/safetensors
#         if criterion == "l1":
#             self.criterion = L1Loss(1.0)
#         elif criterion == "charbonnier":
#             self.criterion = charbonnier_loss(1.0)
#         else:
#             self.criterion = MSSSIML1Loss(1.0)

#     @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
#     def forward(self, sr: Tensor, hr: Tensor, iteration) -> Tensor:
#         # ae_sr = self.ae(sr.clamp(0, 1)).clamp(0, 1)
#         # ae_hr = self.ae(hr.detach()).clamp(0, 1)
#         ae_sr = self.ae(sr)
#         ae_hr = self.ae(hr.detach())

#         loss_debug1 = torch.abs(sr - hr).clone()
#         loss_debug2 = torch.abs(ae_sr - ae_hr).clone()

#         # iteration = self._get_iteration()
#         if iteration % 1000 == 0:
#             self._save_debug_visuals(
#                 sr, hr, ae_sr, ae_hr, loss_debug1, loss_debug2, iteration
#             )

#         return self.loss_weight * self.criterion(ae_sr, ae_hr)
#         # return self.loss_weight * self.criterion(sr, hr)

#     def _get_iteration(self) -> int:
#         """Determine the current iteration based on existing files in the debug directory."""
#         p = os.path.join(self.debug_dir, "loss_l1")
#         if not os.path.exists(p):
#             return 0

#         files = os.listdir(p)
#         iterations = []
#         for file in files:
#             if "_iter_" in file:
#                 try:
#                     iter_num = int(file.split("_iter_")[1].split("_sample_")[0])
#                     iterations.append(iter_num)
#                 except (IndexError, ValueError):
#                     continue

#         return max(iterations, default=0) + 1

#     def _save_debug_visuals(
#         self,
#         pred: Tensor,
#         target: Tensor,
#         pred_ae: Tensor,
#         target_ae: Tensor,
#         loss_l1: Tensor,
#         loss_l1_ae: Tensor,
#         iteration: int,
#     ) -> None:
#         """Save debugging visualizations to disk."""
#         # Detach tensors for visualization
#         pred_vis = pred.detach().cpu()
#         target_vis = target.detach().cpu()
#         pred_vis_ae = pred_ae.detach().cpu()
#         target_vis_ae = target_ae.detach().cpu()
#         loss_vis1 = loss_l1.detach().cpu()
#         loss_vis2 = loss_l1_ae.detach().cpu()

#         os.makedirs(os.path.join(self.debug_dir, "pred"), exist_ok=True)
#         os.makedirs(os.path.join(self.debug_dir, "target"), exist_ok=True)
#         os.makedirs(os.path.join(self.debug_dir, "pred_ae"), exist_ok=True)
#         os.makedirs(os.path.join(self.debug_dir, "target_ae"), exist_ok=True)
#         os.makedirs(os.path.join(self.debug_dir, "loss_l1"), exist_ok=True)
#         os.makedirs(os.path.join(self.debug_dir, "loss_l1_ae"), exist_ok=True)

#         # Save images for the first sample in the batch
#         for i in range(min(pred_vis.shape[0], 1)):
#             save_image(
#                 pred_vis[i],
#                 os.path.join(
#                     self.debug_dir, "pred", f"pred_iter_{iteration}_sample_{i}.png"
#                 ),
#             )
#             save_image(
#                 target_vis[i],
#                 os.path.join(
#                     self.debug_dir, "target", f"target_iter_{iteration}_sample_{i}.png"
#                 ),
#             )

#             save_image(
#                 pred_vis_ae[i],
#                 os.path.join(
#                     self.debug_dir,
#                     "pred_ae",
#                     f"pred_ae_iter_{iteration}_sample_{i}.png",
#                 ),
#             )
#             save_image(
#                 target_vis_ae[i],
#                 os.path.join(
#                     self.debug_dir,
#                     "target_ae",
#                     f"target_ae_iter_{iteration}_sample_{i}.png",
#                 ),
#             )

#             self._save_heatmap(
#                 loss_vis1[i].mean(0),
#                 os.path.join(
#                     self.debug_dir,
#                     "loss_l1",
#                     f"loss_l1_iter_{iteration}_sample_{i}.png",
#                 ),
#             )

#             self._save_heatmap(
#                 loss_vis2[i].mean(0),
#                 os.path.join(
#                     self.debug_dir,
#                     "loss_l1_ae",
#                     f"loss_l1_ae_iter_{iteration}_sample_{i}.png",
#                 ),
#             )

#     def _save_heatmap(self, tensor: Tensor, path: str) -> None:
#         """Save a heatmap for a tensor."""
#         plt.figure(figsize=(6, 6))
#         plt.imshow(tensor.squeeze(), cmap="viridis")
#         plt.colorbar()
#         plt.savefig(path)
#         plt.close()
