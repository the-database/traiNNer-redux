import copy
import os
import time
from collections.abc import Sequence
from logging import Logger
from os import path as osp
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import onnxruntime as ort
import torch
from modelopt.onnx.autocast import convert_to_mixed_precision
from modelopt.onnx.autocast.nodeclassifier import NodeRuleBase
from onnx import ModelProto, TensorProto
from onnxconverter_common.float16 import convert_float_to_float16
from onnxslim import slim
from rich.traceback import install
from torch import Tensor
from torch.export.dynamic_shapes import Dim
from traiNNer.archs.arch_info import REQUIRE_32_HW, REQUIRE_64_HW
from traiNNer.models import build_model
from traiNNer.models.base_model import BaseModel
from traiNNer.utils.config import Config
from traiNNer.utils.logger import clickable_file_path, get_root_logger
from traiNNer.utils.misc import format_duration_min_sec
from traiNNer.utils.redux_options import ReduxOptions

MAX_LEGACY_OPSET = 20
MIN_DYNAMO_OPSET = 18

INPUT_NAME = "input"
OUTPUT_NAME = "output"

DType = Literal["fp32", "fp16", "bf16"]
IODType = Literal["auto", "fp32", "fp16"]

DTYPE_MAP: dict[DType, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

ONNX_DTYPE_MAP: dict[DType, int] = {
    "fp32": TensorProto.FLOAT,
    "fp16": TensorProto.FLOAT16,
    "bf16": TensorProto.BFLOAT16,
}

MODELOPT_PRECISION_MAP: dict[DType, str] = {
    "fp16": "fp16",
    "bf16": "bf16",
}


class VideoModelExportWrapper(torch.nn.Module):
    """
    Wrapper for video models to handle input shape transformation for ONNX export.

    VapourSynth ORT plugin only supports 4D tensors, so this wrapper takes
    4D input (B, T*C, H, W) and reshapes it to 5D (B, T, C, H, W) for the model.
    """

    def __init__(
        self, model: torch.nn.Module, num_frames: int, channels: int = 3
    ) -> None:
        super().__init__()
        self.model = model
        self.num_frames = num_frames
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (B, T*C, H, W) where T*C = num_frames * channels
        Returns:
            Output from the model
        """
        b, _, h, w = x.shape
        # Reshape from (B, T*C, H, W) to (B, T, C, H, W)
        x = x.view(b, self.num_frames, self.channels, h, w)
        return self.model(x)


def get_out_path(
    out_dir: str,
    name: str,
    opset: int,
    dtype: DType,
    optimized: bool,
    dynamo: bool,
    shape: Sequence[int],
    dynamic_flags: Sequence[bool],
    io_dtype: IODType = "auto",
) -> str:
    axis_labels = ["N", "C", "H", "W"]
    if len(shape) == 5:
        axis_labels.insert(1, "T")
    shape_parts = []
    dynamic_dims = []

    for i, (dim_val, is_dyn) in enumerate(zip(shape, dynamic_flags, strict=False)):
        if is_dyn:
            shape_parts.append(axis_labels[i])
            dynamic_dims.append(axis_labels[i])
        else:
            shape_parts.append(str(dim_val))

    shape_str = "x".join(shape_parts)
    if dynamic_dims:
        shape_str += f"_dyn-{''.join(dynamic_dims)}"
    else:
        shape_str += "_static"

    dtype_str = str(dtype)
    if dtype != "fp32":
        dtype_str = f"strong_{dtype}"
    if io_dtype != "auto":
        dtype_str += f"_io-{io_dtype}"

    filename = f"{name}_{shape_str}_{dtype_str}_op{opset}{'_onnxslim' if optimized else ''}{'_dynamo' if dynamo else ''}.onnx"
    return osp.normpath(osp.join(out_dir, filename))


def parse_input_shape(
    shape_str: str,
    net_g_type: str,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    parts = [p.strip() for p in shape_str.lower().split("x")]
    if len(parts) not in (4, 5):
        raise ValueError(
            f"Invalid onnx.shape (expected NxCxHxW or NxTxCxHxW): {shape_str!r}"
        )

    default_hw = 16
    if net_g_type in REQUIRE_32_HW:
        default_hw = 32
    elif net_g_type in REQUIRE_64_HW:
        default_hw = 64

    if len(parts) == 4:
        # NCHW
        defaults = [1, 3, default_hw, default_hw]
    else:
        # NTCHW
        defaults = [1, 1, 3, default_hw, default_hw]

    dims: list[int] = []
    dynamic_flags: list[bool] = []
    for i, p in enumerate(parts):
        if p.isdigit():
            dims.append(int(p))
            dynamic_flags.append(False)
        else:
            dims.append(defaults[i])
            dynamic_flags.append(True)

    return tuple(dims), tuple(dynamic_flags)


class InvalidLowPrecisionGraphError(Exception):
    """The fp32 -> low precision conversion produced a graph that fails strict
    ONNX type checking (e.g. shape-derived scalars cast to fp32 next to fp16
    tensors), which TensorRT and ONNX Runtime reject."""


def validate_low_precision(model: ModelProto) -> None:
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as e:
        raise InvalidLowPrecisionGraphError(str(e)) from e


def ort_basic_optimize(path: str, logger: Logger) -> None:
    """Rewrite the ONNX at path with ONNX Runtime's basic graph optimizations."""
    pre_nodes = len(onnx.load(path).graph.node)
    logger.info(
        "Legacy ONNX conversion complete. Nodes before ORT optimize: %d",
        pre_nodes,
    )

    ort_optimized_path = path + ".ortopt"
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = ort_optimized_path

    logger.info("Optimizing legacy ONNX with ONNX Runtime (ORT_ENABLE_BASIC)...")
    ort.InferenceSession(
        path,
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )

    model_proto_post = onnx.load(ort_optimized_path)
    logger.info(
        "ORT optimization complete. Nodes after ORT optimize: %d",
        len(model_proto_post.graph.node),
    )
    onnx.save(model_proto_post, path)

    try:
        if osp.exists(ort_optimized_path):
            os.remove(ort_optimized_path)
    except OSError:
        pass


def wrap_bf16_io_as_fp16(model: ModelProto) -> ModelProto:
    """Expose a bf16 graph's inputs and outputs as fp16, leaving the interior bf16.

    Inference runtimes that feed fp16 buffers (RGBH clips, fp16 device tensors)
    can then run a strongly typed bf16 engine directly.
    """
    graph = model.graph
    bf16 = TensorProto.BFLOAT16
    fp16 = TensorProto.FLOAT16
    initializer_names = {t.name for t in graph.initializer}

    def is_bf16(value_info: onnx.ValueInfoProto) -> bool:
        return value_info.type.tensor_type.elem_type == bf16

    head: list[onnx.NodeProto] = []
    tail: list[onnx.NodeProto] = []

    for graph_input in graph.input:
        if graph_input.name in initializer_names or not is_bf16(graph_input):
            continue
        external = graph_input.name
        internal = f"{external}_bf16"
        for node in graph.node:
            for i, name in enumerate(node.input):
                if name == external:
                    node.input[i] = internal
        head.append(
            onnx.helper.make_node(
                "Cast", [external], [internal], name=f"{external}_to_bf16", to=bf16
            )
        )
        graph_input.type.tensor_type.elem_type = fp16

    for graph_output in graph.output:
        if not is_bf16(graph_output):
            continue
        external = graph_output.name
        internal = f"{external}_bf16"
        for node in graph.node:
            for i, name in enumerate(node.output):
                if name == external:
                    node.output[i] = internal
            for i, name in enumerate(node.input):
                if name == external:
                    node.input[i] = internal
        tail.append(
            onnx.helper.make_node(
                "Cast", [internal], [external], name=f"{external}_to_fp16", to=fp16
            )
        )
        graph_output.type.tensor_type.elem_type = fp16

    boundary = {vi.name for vi in graph.input} | {vi.name for vi in graph.output}
    stale = [vi for vi in graph.value_info if vi.name in boundary]
    for vi in stale:
        graph.value_info.remove(vi)

    nodes = head + list(graph.node) + tail
    del graph.node[:]
    graph.node.extend(nodes)
    return model


def convert_onnx_to_low_precision(
    onnx_path: str,
    bf16_exclude_depthwise: bool,
    dtype: DType,
    opset: int,
    io_dtype: IODType = "auto",
) -> ModelProto:
    if dtype == "fp32":
        if io_dtype == "fp16":
            raise ValueError("io_dtype: fp16 requires dtype fp16 or bf16")
        return onnx.load(onnx_path)

    # auto keeps the historical behavior: fp16 models get fp16 I/O, bf16 models
    # keep fp32 I/O.
    if io_dtype == "auto":
        keep_io_types = dtype == "bf16"
    else:
        keep_io_types = io_dtype == "fp32"

    if dtype == "fp16":
        torch_dtype = DTYPE_MAP[dtype]
        max_val = torch.finfo(torch_dtype).max
    else:
        max_val = np.inf

    custom_rule = None
    if dtype == "bf16" and bf16_exclude_depthwise:
        fp32_model = onnx.load(onnx_path)
        init_map = {t.name: t for t in fp32_model.graph.initializer}
        custom_rule = SkipDepthwiseConvRule(init_map)
    model = None

    try:
        model = convert_to_mixed_precision(
            onnx_path=onnx_path,
            low_precision_type=MODELOPT_PRECISION_MAP[dtype],
            keep_io_types=keep_io_types,
            data_max=max_val,
            init_max=max_val,
            custom_rule=custom_rule,
            opset=opset,
            op_types_to_exclude=["ConvTranspose"],
        )
        validate_low_precision(model)
    except:  # noqa: E722
        if dtype == "fp16":
            logger = get_root_logger()
            logger.warning(
                "Failed to convert to fp16 with NVIDIA Model Optimizer, falling back to legacy fp16 conversion."
            )
            model = onnx.load(onnx_path)
            model = convert_float_to_float16(model, keep_io_types=keep_io_types)
            validate_low_precision(model)
        else:
            raise

    assert model is not None
    if dtype == "bf16" and io_dtype == "fp16":
        model = wrap_bf16_io_as_fp16(model)

    return model


def convert_and_save_onnx(
    model: BaseModel,
    logger: Logger,
    opt: ReduxOptions,
    torch_input: Tensor,
    dynamic_flags: Sequence[bool],
    out_dir: str,
    example_shape: Sequence[int],
    dtype: DType,
) -> tuple[ModelProto, int, str, str | None]:
    assert model.net_g is not None
    assert opt.onnx is not None

    is_dynamo = bool(opt.onnx.dynamo)
    requested_opset = opt.onnx.opset

    # Check if we should wrap 5D input to 4D for vsmlrt compatibility
    wrap_5d = opt.onnx.wrap_5d_to_4d and len(example_shape) == 5

    if wrap_5d:
        n, t, c, h, w = example_shape
        # Create 4D shape: (N, T*C, H, W)
        export_shape: Sequence[int] = (n, t * c, h, w)
        # Create 4D input tensor
        export_input = torch_input.view(n, t * c, h, w)
        # Wrap the model
        export_model: torch.nn.Module = VideoModelExportWrapper(
            model.net_g, num_frames=t, channels=c
        )
        # Convert 5D dynamic flags to 4D: (N, T, C, H, W) -> (N, T*C, H, W)
        # T*C channel dim is static (T and C must both be static for the reshape to work)
        export_dynamic_flags: Sequence[bool] = (
            dynamic_flags[0],  # N
            False,  # T*C (merged, must be static)
            dynamic_flags[3],  # H
            dynamic_flags[4],  # W
        )
        axis_names = ["batch_size", "channels", "height", "width"]
        logger.info(
            "Wrapping 5D VSR model for 4D export (vsmlrt compatibility): %s -> %s",
            "x".join(str(d) for d in example_shape),
            "x".join(str(d) for d in export_shape),
        )
    else:
        export_shape = example_shape
        export_input = torch_input
        export_model = model.net_g
        export_dynamic_flags = dynamic_flags
        axis_names = ["batch_size", "channels", "height", "width"]
        if len(example_shape) == 5:
            axis_names.insert(1, "temporal")

    has_dynamic = any(export_dynamic_flags)

    if not has_dynamic:
        logger.info(
            "Exporting ONNX with fully static input shape %s (N,C,H,W), dtype=%s.",
            opt.onnx.shape,
            dtype,
        )
    else:
        dynamic_dims = [
            name
            for name, is_dyn in zip(axis_names, export_dynamic_flags, strict=False)
            if is_dyn
        ]
        static_dims = [
            name
            for name, is_dyn in zip(axis_names, export_dynamic_flags, strict=False)
            if not is_dyn
        ]
        logger.info(
            "Exporting ONNX with mixed static/dynamic input shape %s (N,C,H,W). "
            "Example shape: %s. Dynamic dims: %s. Static dims: %s. dtype=%s.",
            opt.onnx.shape,
            "x".join(str(d) for d in export_shape),
            ", ".join(dynamic_dims),
            ", ".join(static_dims),
            dtype,
        )

    if not has_dynamic:
        dynamic_shapes = None
        dynamic_axes = None
    else:
        dim_specs = [
            Dim.AUTO if is_dyn else Dim.STATIC for is_dyn in export_dynamic_flags
        ]
        dynamic_shapes = (tuple(dim_specs),)

        dynamic_axes = {INPUT_NAME: {}, OUTPUT_NAME: {}}
        for axis, is_dyn in enumerate(export_dynamic_flags):
            if not is_dyn:
                continue
            name = axis_names[axis]
            dynamic_axes[INPUT_NAME][axis] = name
            dynamic_axes[OUTPUT_NAME][axis] = name

    if is_dynamo:
        if requested_opset < MIN_DYNAMO_OPSET:
            logger.info(
                (
                    "Requested ONNX opset %d is below the minimum for dynamo export (%d). "
                    "Using %d instead. If you really need opset %d, try disabling dynamo "
                    "with dynamo: false in your config."
                ),
                requested_opset,
                MIN_DYNAMO_OPSET,
                MIN_DYNAMO_OPSET,
                requested_opset,
            )
        opset = max(MIN_DYNAMO_OPSET, requested_opset)

        dynamic_axes = None
    else:
        if requested_opset > MAX_LEGACY_OPSET:
            logger.info(
                (
                    "Requested ONNX opset %d is above the maximum for legacy export (%d). "
                    "Using %d instead. If you really need opset %d, try enabling dynamo "
                    "with dynamo: true in your config."
                ),
                requested_opset,
                MAX_LEGACY_OPSET,
                MAX_LEGACY_OPSET,
                requested_opset,
            )
        opset = min(MAX_LEGACY_OPSET, requested_opset)

        dynamic_shapes = None

    fp32_out_path = get_out_path(
        out_dir,
        opt.name,
        opset,
        dtype="fp32",
        optimized=False,
        dynamo=is_dynamo,
        shape=export_shape,
        dynamic_flags=export_dynamic_flags,
    )

    if dtype == "fp32":
        temp_out_path = fp32_out_path
    else:
        temp_out_path = fp32_out_path + ".export_temp"

    out_path = get_out_path(
        out_dir,
        opt.name,
        requested_opset,
        dtype=dtype,
        optimized=False,
        dynamo=is_dynamo,
        shape=export_shape,
        dynamic_flags=export_dynamic_flags,
        io_dtype=opt.onnx.io_dtype,
    )

    with torch.inference_mode():
        onnx_program = torch.onnx.export(
            export_model,
            (export_input,),
            None if is_dynamo else temp_out_path,
            dynamo=is_dynamo,
            verbose=False,
            optimize=False,
            opset_version=opset,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_shapes=dynamic_shapes,
            dynamic_axes=dynamic_axes,
            verify=opt.onnx.verify if dtype == "fp32" else False,
        )

        if is_dynamo:
            assert onnx_program is not None
            logger.info("Dynamo ONNX conversion complete. Optimizing...")

            pre_nodes = len(onnx_program.model.graph)
            logger.info("Dynamo export nodes before optimize(): %d", pre_nodes)

            onnx_program.optimize()
            onnx_program.save(temp_out_path)

            post_nodes = len(onnx_program.model.graph)
            logger.info("Dynamo export nodes after optimize(): %d", post_nodes)

        else:
            ort_basic_optimize(temp_out_path, logger)

    fp32_saved_path: str | None = None

    if dtype != "fp32":
        fp32_model_proto = onnx.load(temp_out_path)
        onnx.save(fp32_model_proto, fp32_out_path)
        logger.info(
            "Saved FP32 model to %s",
            clickable_file_path(Path(fp32_out_path).absolute().parent, fp32_out_path),
        )
        fp32_saved_path = fp32_out_path

        logger.info(
            "Converting ONNX model to %s using NVIDIA Model Optimizer...", dtype
        )
        try:
            model_proto = convert_onnx_to_low_precision(
                temp_out_path,
                opt.onnx.bf16_exclude_depthwise,
                dtype,
                requested_opset,
                opt.onnx.io_dtype,
            )
        except InvalidLowPrecisionGraphError as e:
            if dtype != "fp16" or is_dynamo or opt.onnx.io_dtype == "fp32":
                raise
            # Graph converters type shape-derived scalars (e.g. a token count
            # that overflows fp16) inconsistently. Tracing the model in half
            # precision instead keeps that scalar math in fp32 and casts at
            # the point of use, as PyTorch itself does.
            logger.warning(
                "fp16 graph conversion produced an invalid graph (%s); "
                "exporting the model directly in half precision instead.",
                str(e).splitlines()[0],
            )
            half_model = copy.deepcopy(export_model).half()
            with torch.inference_mode():
                torch.onnx.export(
                    half_model,
                    (export_input.half(),),
                    temp_out_path,
                    dynamo=False,
                    verbose=False,
                    opset_version=opset,
                    input_names=[INPUT_NAME],
                    output_names=[OUTPUT_NAME],
                    dynamic_axes=dynamic_axes,
                )
            del half_model
            # No ORT optimize pass here: on the CPU EP it inserts fp32 casts
            # around ops lacking fp16 kernels and bakes them into the graph
            # (measured 23 -> 39 ms per frame for TFDAT under TensorRT).
            model_proto = onnx.load(temp_out_path)
            validate_low_precision(model_proto)
        onnx.save(model_proto, out_path)

        if osp.exists(temp_out_path):
            os.remove(temp_out_path)
    else:
        model_proto = onnx.load(temp_out_path)

    assert model_proto is not None
    return model_proto, opset, out_path, fp32_saved_path


def verify_onnx(
    model: BaseModel,
    logger: Logger,
    torch_input: Tensor,
    onnx_path: str,
    dtype: DType,
    wrap_5d: bool = False,
) -> None:
    assert model.net_g is not None

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    if dtype == "bf16":
        logger.info(
            "Skipping ONNX Runtime verification for bf16 model (ORT CUDA bf16 unsupported/limited). "
            "Use TensorRT with --stronglyTyped for inference."
        )
        return

    # Handle wrapped 5D -> 4D models
    if wrap_5d and torch_input.dim() == 5:
        n, t, c, h, w = torch_input.shape
        verify_model: torch.nn.Module = VideoModelExportWrapper(
            model.net_g, num_frames=t, channels=c
        )
        verify_input = torch_input.view(n, t * c, h, w)
    else:
        verify_model = model.net_g
        verify_input = torch_input

    with torch.inference_mode():
        if dtype == "fp16":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                torch_output_np = verify_model(verify_input).float().cpu().numpy()
        else:
            torch_output_np = verify_model(verify_input).cpu().numpy()

    providers = ["CUDAExecutionProvider"]

    try:
        ort_session = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        logger.warning(
            "Could not create ONNX Runtime session for verification (dtype=%s): %s",
            dtype,
            e,
        )
        return

    ort_input = ort_session.get_inputs()[0]
    input_np = (
        verify_input.cpu().numpy().astype(np.float16)
        if ort_input.type == "tensor(float16)"
        else verify_input.cpu().numpy()
    )

    ort_inputs = {ort_input.name: input_np}

    try:
        onnx_output = ort_session.run(None, ort_inputs)
    except Exception as e:
        logger.warning("ONNX Runtime inference failed during verification: %s", e)
        return

    onnx_output_fp32 = (
        onnx_output[0].astype(np.float32)
        if onnx_output[0].dtype != np.float32
        else onnx_output[0]
    )

    if dtype == "fp16":
        rtol, atol = 1e-02, 1e-02
    else:
        rtol, atol = 1e-02, 1e-03

    try:
        np.testing.assert_allclose(
            torch_output_np,
            onnx_output_fp32,
            rtol=rtol,
            atol=atol,
        )
        logger.info(
            "ONNX output verified against PyTorch output successfully (dtype=%s).",
            dtype,
        )
    except AssertionError as e:
        logger.warning(
            "ONNX verification completed with warnings (dtype=%s): %s", dtype, e
        )


def convert_pipeline(root_path: str) -> None:
    install()
    torch.cuda.set_per_process_memory_fraction(fraction=1.0)
    opt, _ = Config.load_config_from_file(root_path, is_train=False)
    model = build_model(opt)
    assert opt.onnx is not None
    assert opt.network_g is not None

    dtype: DType = getattr(opt.onnx, "dtype", "fp32")
    if dtype not in DTYPE_MAP:
        raise ValueError(
            f"Invalid dtype '{dtype}'. Must be one of: {list(DTYPE_MAP.keys())}"
        )

    example_shape, dynamic_flags = parse_input_shape(
        opt.onnx.shape, opt.network_g["type"]
    )

    wrap_5d = opt.onnx.wrap_5d_to_4d and len(example_shape) == 5

    torch_input = torch.randn(*example_shape, device="cuda", dtype=torch.float32)

    start_time = time.time()

    assert model.net_g is not None
    model.net_g.eval()
    logger = get_root_logger()

    out_dir = "./onnx"
    os.makedirs(out_dir, exist_ok=True)

    model_proto, opset, out_path, fp32_path = convert_and_save_onnx(
        model, logger, opt, torch_input, dynamic_flags, out_dir, example_shape, dtype
    )

    end_time = time.time()

    logger.info(
        "Saved to %s in %s.",
        clickable_file_path(Path(out_path).absolute().parent, out_path),
        format_duration_min_sec(end_time - start_time),
    )

    if not opt.onnx.dynamo:
        if opt.onnx.verify:
            verify_onnx(model, logger, torch_input, out_path, dtype, wrap_5d)
            if fp32_path is not None:
                verify_onnx(model, logger, torch_input, fp32_path, "fp32", wrap_5d)

    if opt.onnx.optimize:
        logger.info("Optimizing ONNX with OnnxSlim...")
        model_proto = slim(
            model_proto,
        )

        assert isinstance(model_proto, ModelProto)

        # Use export_shape for optimized path (4D if wrapped, original otherwise)
        if wrap_5d:
            n, t, c, h, w = example_shape
            opt_shape: Sequence[int] = (n, t * c, h, w)
            opt_dynamic_flags: Sequence[bool] = (
                dynamic_flags[0],
                False,
                dynamic_flags[3],
                dynamic_flags[4],
            )
        else:
            opt_shape = example_shape
            opt_dynamic_flags = dynamic_flags

        optimized_path = get_out_path(
            out_dir,
            opt.name,
            opset,
            dtype=dtype,
            optimized=True,
            dynamo=opt.onnx.dynamo,
            shape=opt_shape,
            dynamic_flags=opt_dynamic_flags,
            io_dtype=opt.onnx.io_dtype,
        )

        session_opt = ort.SessionOptions()
        session_opt.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        session_opt.optimized_model_filepath = optimized_path

        providers = ["CUDAExecutionProvider"]

        try:
            ort.InferenceSession(out_path, session_opt, providers=providers)
            if opt.onnx.verify:
                verify_onnx(model, logger, torch_input, optimized_path, dtype, wrap_5d)
            model_proto = onnx.load(optimized_path)
            logger.info(
                "Optimized model saved to %s",
                clickable_file_path(
                    Path(optimized_path).absolute().parent, optimized_path
                ),
            )
        except Exception as e:
            logger.warning("ONNX optimization failed: %s", e)


class SkipDepthwiseConvRule(NodeRuleBase):
    def __init__(
        self,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> None:
        self.inits = initializer_map
        self.logger = get_root_logger()

    def _check_inner(self, node: onnx.NodeProto) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        if node.op_type != "Conv":
            return False

        group = 1
        for a in node.attribute:
            if a.name == "group":
                group = int(onnx.helper.get_attribute_value(a))
                break
        if group <= 1:
            return False

        if len(node.input) < 2:
            return False
        w_name = node.input[1]
        w_init = self.inits.get(w_name)
        if w_init is None:
            return False

        w = onnx.numpy_helper.to_array(w_init)
        if w.ndim < 3:
            return False

        cout = int(w.shape[0])
        cin_per_group = int(w.shape[1])

        return (cin_per_group == 1) and (group == cout)

    def _log_skipped(self, node: onnx.NodeProto, **kwargs) -> None:
        self.logger.info("Skipping depthwise Conv node %s (kept FP32)", node.name)


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    convert_pipeline(root_path)
