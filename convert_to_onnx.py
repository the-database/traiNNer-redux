import os
import time
from logging import Logger
from os import path as osp
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import ModelProto
from onnxconverter_common.float16 import convert_float_to_float16
from onnxslim import slim
from rich.traceback import install
from torch import Tensor
from torch.export.dynamic_shapes import Dim
from traiNNer.models import build_model
from traiNNer.models.base_model import BaseModel
from traiNNer.utils.config import Config
from traiNNer.utils.logger import clickable_file_path, get_root_logger
from traiNNer.utils.redux_options import ReduxOptions

MAX_LEGACY_OPSET = 20
MIN_DYNAMO_OPSET = 18


def get_out_path(
    out_dir: str,
    name: str,
    opset: int,
    fp16: bool,
    optimized: bool,
    dynamo: bool,
) -> str:
    filename = f"{name}_fp{'16' if fp16 else '32'}_op{opset}{'_onnxslim' if optimized else ''}{'_dynamo' if dynamo else ''}.onnx"
    return osp.normpath(osp.join(out_dir, filename))


def convert_and_save_onnx(
    model: BaseModel,
    logger: Logger,
    opt: ReduxOptions,
    torch_input: Tensor,
    out_dir: str,
) -> tuple[ModelProto, int, str]:
    assert model.net_g is not None
    assert opt.onnx is not None

    is_dynamo = bool(opt.onnx.dynamo)
    requested_opset = opt.onnx.opset

    if opt.onnx.use_static_shapes:
        input_names: list[str] | None = None
        output_names: list[str] | None = None
        dynamic_shapes = None
        dynamic_axes = None
    else:
        input_names = ["input"]
        output_names = ["output"]

        dynamic_shapes = ((Dim.AUTO, Dim.STATIC, Dim.AUTO, Dim.AUTO),)

        dynamic_axes = {
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        }

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

    out_path = get_out_path(
        out_dir,
        opt.name,
        opset,
        fp16=False,
        optimized=False,
        dynamo=is_dynamo,
    )

    with torch.inference_mode():
        onnx_program = torch.onnx.export(
            model.net_g,
            (torch_input,),
            None if is_dynamo else out_path,
            dynamo=is_dynamo,
            verbose=False,
            optimize=False,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            dynamic_axes=dynamic_axes,
            verify=opt.onnx.verify,
        )

        if is_dynamo:
            assert onnx_program is not None
            logger.info("Dynamo ONNX conversion complete. Optimizing...")
            onnx_program.optimize()
            onnx_program.save(out_path)

    model_proto = onnx.load(out_path)
    assert model_proto is not None
    return model_proto, opset, out_path


def verify_onnx(
    model: BaseModel, logger: Logger, torch_input: Tensor, onnx_path: str
) -> None:
    assert model.net_g is not None

    with torch.inference_mode():
        torch_output_np = model.net_g(torch_input).cpu().numpy()

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    ort_inputs = {ort_session.get_inputs()[0].name: torch_input.cpu().numpy()}
    onnx_output = ort_session.run(None, ort_inputs)

    try:
        np.testing.assert_allclose(
            torch_output_np,
            onnx_output[0],  # pyright: ignore
            rtol=1e-02,
            atol=1e-03,
        )
        logger.info("ONNX output verified against PyTorch output successfully.")
    except AssertionError as e:
        logger.warning("ONNX verification completed with warnings: %s", e)


def convert_pipeline(root_path: str) -> None:
    install()
    torch.cuda.set_per_process_memory_fraction(fraction=1.0)
    opt, _ = Config.load_config_from_file(root_path, is_train=False)
    model = build_model(opt)
    assert opt.onnx is not None

    if opt.onnx.use_static_shapes:
        dims = tuple(map(int, opt.onnx.shape.split("x")))
        torch_input = torch.randn(*dims, device="cuda")
    else:
        torch_input = torch.randn(1, 3, 32, 32, device="cuda")
    start_time = time.time()

    assert model.net_g is not None
    model.net_g.eval()
    logger = get_root_logger()

    out_dir = "./onnx"
    os.makedirs(out_dir, exist_ok=True)

    model_proto, opset, out_path_fp32 = convert_and_save_onnx(
        model, logger, opt, torch_input, out_dir
    )

    end_time = time.time()

    logger.info(
        "Saved to %s in %.2f seconds.",
        clickable_file_path(Path(out_path_fp32).absolute().parent, out_path_fp32),
        end_time - start_time,
    )

    if not opt.onnx.dynamo:
        if opt.onnx.verify:
            verify_onnx(model, logger, torch_input, out_path_fp32)

    if opt.onnx.optimize:
        logger.info("Optimizing ONNX with OnnxSlim...")
        model_proto = slim(
            model_proto,
        )

        assert isinstance(model_proto, ModelProto)

        session_opt = ort.SessionOptions()
        session_opt.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        session_opt.optimized_model_filepath = get_out_path(
            out_dir, opt.name, opset, fp16=False, optimized=True, dynamo=opt.onnx.dynamo
        )
        ort.InferenceSession(out_path_fp32, session_opt)
        verify_onnx(model, logger, torch_input, session_opt.optimized_model_filepath)
        model_proto = onnx.load(session_opt.optimized_model_filepath)

    if opt.onnx.fp16:
        start_time = time.time()
        out_path = get_out_path(
            out_dir, opt.name, opset, True, opt.onnx.optimize, opt.onnx.dynamo
        )
        model_proto_fp16 = convert_float_to_float16(model_proto)
        onnx.save(model_proto_fp16, out_path)
        end_time = time.time()
        logger.info(
            "Saved to %s in %.2f seconds.",
            clickable_file_path(Path(out_path).absolute().parent, out_path),
            end_time - start_time,
        )


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    convert_pipeline(root_path)
