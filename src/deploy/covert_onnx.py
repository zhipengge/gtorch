# -*- coding: utf-8 -*-
"""
@author: gehipeng @ 20231005
@file: covert_onnx.py
@brief: covert_onnx
"""
import torch
import torch.nn as nn


def export_onnx(model, inputs, input_names, output_names, save_path):
    model.eval()
    torch.onnx.export(
        model=model,  # nn.Module instance
        args=inputs,  # input tensors
        f=save_path,  # file path to export
        input_names=input_names,  # input tensor names
        output_names=output_names,  # output tensor names
        dynamic_axes={"input0": {0: "batch"}, "output0": {0: "batch"}},  # dynamic size
        opset_version=12,  # operation set version
    )
    print("Export onnx done.")


if __name__ == "__main__":

    class Model(nn.Module):
        def __init__(self, in_features, out_features) -> None:
            super().__init__()
            self.linear1 = nn.Linear(in_features, out_features)
            self.linear2 = nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.linear1(x), self.linear2(x)

    model = Model(4, 3)
    inputs = (torch.rand([1, 2, 3, 4]),)
    input_names = ("input0",)
    output_names = ("output0", "output1")
    import os

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../output"
    )
    save_path = os.path.join(output_dir, "example.onnx")
    export_onnx(model, inputs, input_names, output_names, save_path)
