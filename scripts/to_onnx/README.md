# Export ONNX

The following procedures were used to generate our ONNX models.

1. Install dependencies
```sh
pip install onnx
pip install onnxruntime
```

2. Use the export script. Our export script only support opset 11 and up. The input shape of the current code is [1, 3, 640, 960].
If you want to support inputs with other resolution, you must adapt the code yourself.
**Please note that the input resolution must be divisible by 8.**
```sh
python convert_onnx.py 
```
