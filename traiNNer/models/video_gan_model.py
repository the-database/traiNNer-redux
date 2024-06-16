from ..utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from .video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class VideoGANModel(SRModel, VideoBaseModel):
    """Video GAN model.

    Use multiple inheritance.
    It will first use the functions of :class:`SRGANModel`:

    - :func:`init_training_settings`
    - :func:`setup_optimizers`
    - :func:`optimize_parameters`
    - :func:`save`

    Then find functions in :class:`VideoBaseModel`.
    """
