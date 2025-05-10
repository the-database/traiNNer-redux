import os
import time
from logging import Logger
from os import path as osp
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import ModelProto
from onnxconverter_common.float16 import convert_float_to_float16
from onnxslim import slim
from rich.traceback import install
from torch import Tensor
from traiNNer.models import build_model
from traiNNer.models.base_model import BaseModel
from traiNNer.utils.config import Config
from traiNNer.utils.logger import clickable_file_path, get_root_logger
from traiNNer.utils.redux_options import ReduxOptions


def get_out_path(
    out_dir: str, name: str, opset: int, fp16: bool = False, optimized: bool = False
) -> str:
    filename = f"{name}_fp{'16' if fp16 else '32'}_op{opset}{'_onnxslim' if optimized else ''}.onnx"
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

    if opt.onnx.use_static_shapes:
        dynamic_axes = None
        input_names = None
        output_names = None
    else:
        dynamic_axes = {
            "input": {0: "batch_size", 2: "width", 3: "height"},
            "output": {0: "batch_size", 2: "width", 3: "height"},
        }
        input_names = ["input"]
        output_names = ["output"]

    out_path = get_out_path(out_dir, opt.name, opt.onnx.opset, False)

    torch.onnx.export(
        model.net_g,
        (torch_input,),
        out_path,
        dynamo=opt.onnx.dynamo,
        verbose=False,
        opset_version=opt.onnx.opset,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
        output_names=output_names,
    )
    model_proto = onnx.load(out_path)

    assert model_proto is not None

    return model_proto, opt.onnx.opset, out_path


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
        np.testing.assert_allclose(  # pyright: ignore # TODO onnx 1.18
            torch_output_np,
            onnx_output[0],
            rtol=1e-02,
            atol=1e-03,  # pyright: ignore # TODO onnx 1.18
        )
        logger.info("ONNX output verified against PyTorch output successfully.")
    except AssertionError as e:
        logger.warning("ONNX verification completed with warnings: %s", e)


def convert_pipeline(root_path: str) -> None:
    install()
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

    if opt.onnx.verify:
        verify_onnx(model, logger, torch_input, out_path_fp32)

    if opt.onnx.optimize:
        model_proto = slim(
            model_proto,
        )

        assert isinstance(model_proto, ModelProto)

        session_opt = ort.SessionOptions()
        session_opt.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        session_opt.optimized_model_filepath = get_out_path(
            out_dir, opt.name, opset, False, True
        )
        ort.InferenceSession(out_path_fp32, session_opt)
        verify_onnx(model, logger, torch_input, session_opt.optimized_model_filepath)
        model_proto = onnx.load(session_opt.optimized_model_filepath)

    if opt.onnx.fp16:
        start_time = time.time()
        out_path = get_out_path(out_dir, opt.name, opset, True, opt.onnx.optimize)
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
