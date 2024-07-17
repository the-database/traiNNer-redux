# TODO
# if __name__ == "__main__":
#     torch_model = dat_2(scale=4).cuda()
#     torch_input = torch.randn(1, 1, 32, 32, device="cuda")
#     start_time = time.time()
#     # onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
#     torch.onnx.export(
#         torch_model, torch_input, "dat_2.onnx", verbose=False, opset_version=17
#     )

#     end_time = time.time()
#     print(f"{end_time - start_time} seconds")
#     # onnx_program.save("dat_2.onnx")
