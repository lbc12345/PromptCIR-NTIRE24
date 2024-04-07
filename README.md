# Blind Compressed Image Restoration with Prompt Learning

This repository is the official PyTorch validation code of PromptCIR: Blind Compressed Image Restoration with Prompt Learning (NTIRE24)

🔥 We are the winner of the "NTIRE 2024 Blind Compressed Image Enhancement Challenge" [track](https://codalab.lisn.upsaclay.fr/competitions/17548)!

## :sparkles: Getting Start
### Preparing Environment

Python == 3.9 \
Pytorch == 1.9.0

```bash
conda create -n PromptCIR python=3.9
conda activate PromptCIR
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Preparing checkpoints
Download checkpoints from [Google Drive](https://drive.google.com/drive/folders/1oxOVDK845rikuwWiMICHUv5nFGJjA7pV?usp=drive_link)

## Evaluation

Run the code
```bash
python test_model.py --data_lq_root /path/to/inputLQ --ckpt_path checkpoint/model_best.pth --ckpt_path2 checkpoint/model_best2.pth --save --data_out_root /path/to/saveimage
```

Please put LQ images in the "/path/to/inputLQ".
