import torch.onnx
import onnx
import argparse
import os
import onnxruntime
import numpy as np

from retrain.LEAStereo import LEAStereo
from config_utils.leastereo_args import add_leastereo_args


def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='leastereo', help='onnx export name')
    parser.add_argument('--cuda', type=bool, default=False, help='use cuda?')
    parser.add_argument('--crop_height', type=int, required=True, help="crop height")
    parser.add_argument('--crop_width', type=int, required=True, help="crop width")
    parser.add_argument('--maxdisp', type=int, default=192, help="max disp")
    parser.add_argument('--resume', type=str, default='', help="resume from saved model")

    add_leastereo_args(parser)

    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()

    device = 'cpu'
    if args.cuda:
        device = 'cuda'

    model = LEAStereo(args, device)
    if args.resume:
        __load_checkpoint(model, device, args.resume)

    model.eval()

    batch_size = 1
    left = torch.randn(batch_size, 3, args.crop_height, args.crop_width)
    right = torch.randn(batch_size, 3, args.crop_height, args.crop_width)

    output = model(left, right)

    torch.onnx.export(model,
                      (left, right),
                      args.name,
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=False,
                      input_names=['left', 'right'],
                      output_names=['output'],
                      dynamic_axes={
                          'left': {0: 'batch_size'},
                          'right': {0: 'batch_size'},
                          'output': {0: 'batch_size'}
                      })


def __validate_onnx_model(args, left, right, orig_output):
    onnx_model = onnx.load(args.name)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(args.name)

    def to_numpy(tensor):
        return tensor.detach().numpy()

    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(left),
        ort_session.get_inputs()[1].name: to_numpy(right)
    }
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"{orig_output.shape}")
    print(f"{ort_outs[0].shape}")

    np.testing.assert_allclose(to_numpy(ort_outs), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def __load_checkpoint(model, device, checkpoint):
    if os.path.isfile(checkpoint):
        print(f"=> loading checkpoint '{checkpoint}'")
        checkpoint = torch.load(checkpoint, map_location=torch.device(device))

        if device == 'cpu':
            state_dict = dict()
            for key in checkpoint['state_dict'].keys():
                unwrapped_key = key.split('.', 1)[1] if key.startswith('module') else key
                state_dict[unwrapped_key] = checkpoint['state_dict'][key]
        else:
            state_dict = checkpoint['state_dict']

        model.load_state_dict(state_dict, strict=True)

    else:
        print(f"=> no checkpoint found at '{checkpoint}'")




