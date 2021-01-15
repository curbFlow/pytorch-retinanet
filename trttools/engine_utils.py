import os
import sys
import time

import tensorrt as trt
import torch

from trttools import common

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


def gpu_warmup(batch, trt_engine_path):
    print('DOING GPU WARMUP')
    batch = torch.from_numpy(batch)
    inference_image = batch.permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
    with get_engine(trt_engine_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.

        print(f'Image shape: {inference_image.cpu().numpy().shape}')
        print(f'Image type: {type(inference_image.cpu().numpy())}')

        inputs[0].host = inference_image.cpu().numpy()
        st = time.time()
        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        print(f'Inference time: {time.time() - st}')
    print('GPU WARMUP COMPLETE')


def get_engine(engine_file_path=''):
    """Loads a serialized engine."""
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print('Reading engine from file {}'.format(engine_file_path))
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print('ENGINE FILE PATH INVALID.')
        sys.exit()
