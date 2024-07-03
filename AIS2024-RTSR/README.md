## Requirements
- Python >= 3.7
- Pytorch >= 1.10
- NVIDIA GPU + CUDA 
- opencv-python ```pip install opencv-python```
- numpy ```pip install numpy```
- glob ```pip install glob```
- avif ````pip install pillow-avif-plugin```


## Quickstart
- Place the test images in ``test_data`` folder. (like ``test_data/<your_image>``)
- Run the following script.
```bash
python inference.py --input ./test_data \
                               --output ./results/ \
                               --model_path ./pretrained_model/vpeg_s.pth
```

- You can find the result images from ```results/``` folder.
