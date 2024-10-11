import time
from os import path as osp

import numpy as np
import torch
from rich.traceback import install
from traiNNer.models import build_model
from traiNNer.utils.config import Config
from traiNNer.utils.logger import get_root_logger


def convert_pipeline(root_path: str) -> None:
    install()
    opt, _ = Config.load_config_from_file(root_path, is_train=False)
    model = build_model(opt)
    assert opt.onnx is not None

    if opt.onnx.use_static_shapes:
        dims = tuple(map(int, opt.onnx.shape.split("x")))
        torch_input = torch.randn(1, *dims, device="cuda")
    else:
        torch_input = torch.randn(1, 3, 32, 32, device="cuda")
    start_time = time.time()

    assert model.net_g is not None
    model.net_g.eval()
    logger = get_root_logger()

    out_path = f"{opt.name}.onnx"
    if opt.onnx.dynamo:
        onnx_program = torch.onnx.dynamo_export(
            model.net_g,
            torch_input,
            export_options=torch.onnx.ExportOptions(
                dynamic_shapes=not opt.onnx.use_static_shapes
            ),
        )
        logger.info(onnx_program.model_proto.graph.input[0])
        onnx_program.save(out_path)
    else:
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

        opset = 17

        if isinstance(opt.onnx.opset, int):
            opset = opt.onnx.opset

        torch.onnx.export(
            model.net_g,
            torch_input,
            out_path,
            verbose=False,
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            output_names=output_names,
        )

    end_time = time.time()
    logger.info("Saved to %s in %.2f seconds.", out_path, end_time - start_time)

    if opt.onnx.verify:
        import onnx
        import onnxruntime as ort

        with torch.inference_mode():
            torch_output_np = model.net_g(torch_input).cpu().numpy()

        onnx_model = onnx.load(out_path)
        onnx.checker.check_model(onnx_model)

        ort_session = ort.InferenceSession(
            out_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        ort_inputs = {ort_session.get_inputs()[0].name: torch_input.cpu().numpy()}
        onnx_output = ort_session.run(None, ort_inputs)

        np.testing.assert_allclose(
            torch_output_np, onnx_output[0], rtol=1e-02, atol=1e-03
        )


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    convert_pipeline(root_path)
