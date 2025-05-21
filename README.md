# Adaptive Correlation Learning for Cross-modal Hashing

### Dependencies

- Python 3.9

- PyTorch 2.1.0


### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Datasets
- MIRFlickr, [Google drive](https://drive.google.com/drive/folders/1HzIur2yqSrnBcW8hUcHOAVrcdOFj8Qsv?usp=sharing), [BaiduPan(password: ege3)](https://pan.baidu.com/s/1M3w7-s0SIpnm1a-3S7TfgA)

### Process
 - Place the datasets in `data/`
 - Train a model:
 ```bash
 python main.py
