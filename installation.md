pip install --ignore-installed blinker

pip install einops timm==0.6.13 PyWavelets basicsr pandas fairscale pytorch_wavelets gdown scikit-learn matplotlib visualdl colored fvcore antialiased_cnns torch_dct seaborn gradio_client gradio openai transformers accelerate qwen_vl_utils

pip install flash-attn --no-build-isolation

!sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /usr/local/lib/python3.12/dist-packages/basicsr/data/degradations.py

!cd /workspace/ultralytics && pip install -e .

!yolo detect train data=data.yaml model=yolo11n epochs=100 imgsz=640 batch=0.8 save=True amp=False

<Option> pip install --force-reinstall --no-cache-dir "setuptools<81"