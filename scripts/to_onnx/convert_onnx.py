import os
import cv2 
import glob
import numpy as np 
import onnx
import onnxruntime as ort
import torch  
import torch.onnx 
from basicsr.archs.safmn_arch import SAFMN


def convert_onnx(model, output_folder, is_dynamic_batches=False): 
    model.eval() 

    fake_x = torch.rand(1, 3, 640, 960, requires_grad=False)
    output_name = os.path.join(output_folder, 'SAFMN_640_960_x2.onnx')
    dynamic_params = None
    if is_dynamic_batches:
        dynamic_params = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

    # Export the model   
    torch.onnx.export(model,            # model being run 
        fake_x,                         # model input (or a tuple for multiple inputs) 
        output_name,                    # where to save the model  
        export_params=True,             # store the trained parameter weights inside the model file 
        opset_version=15,               # the ONNX version to export the model to 
        do_constant_folding=True,       # whether to execute constant folding for optimization 
        input_names = ['input'],        # the model's input names 
        output_names = ['output'],      # the model's output names 
        dynamic_axes=dynamic_params) 

    print('Model has been converted to ONNX')


def convert_pt(model, output_folder): 
    model.eval() 

    fake_x = torch.rand(1, 3, 640, 960, requires_grad=False)
    output_name = os.path.join(output_folder, 'SAFMN_640_960_x2.pt')

    traced_module = torch.jit.trace(model, fake_x)
    traced_module.save(output_name)
    print('Model has been converted to pt')


def test_onnx(onnx_model, input_path, save_path):
    # for GPU inference
    # ort_session = ort.InferenceSession(onnx_model, ['CUDAExecutionProvider'])

    ort_session = ort.InferenceSession(onnx_model)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(input_path, '*')))):
        # read image
        imgname = os.path.splitext(os.path.basename(path))[0]

        print(f'Testing......idx: {idx}, img: {imgname}')

        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    
        if img.size != (960, 640):
            img = cv2.resize(img, (960, 640), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB, HWC -> CHW
        img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = np.ascontiguousarray(img)

        output = ort_session.run(None, {"input": img})

        # save image
        print('Saving!')
        output = np.squeeze(output[0], axis=0)
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            
        output = (output.clip(0, 1) * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(save_path, f'{imgname}_SAFMN.png'), output)


if __name__ == "__main__":
    model = model = SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=2) 

    pretrained_model = 'experiments/pretrained_models/SAFMN_L_Real_LSDIR_x2.pth'
    model.load_state_dict(torch.load(pretrained_model)['params'], strict=True)

    ###################Onnx export#################
    output_folder = 'scripts/convert' 

    convert_onnx(model, output_folder)
    convert_pt(model, output_folder)

    ###################Test the converted model #################
    onnx_model = 'scripts/convert/SAFMN_640_960_x2.onnx'
    input_path = 'datasets/real_test'
    save_path = 'results/onnx_results'
    test_onnx(onnx_model, input_path, save_path)



