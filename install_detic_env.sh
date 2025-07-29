#!/bin/bash

echo "✅ conda 환경 생성 중..."
conda create -n detic_env python=3.9 -y || exit 1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate detic_env

echo "✅ PyTorch 1.13.1 + CUDA 11.6 설치 중..."
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
  -f https://download.pytorch.org/whl/cu116/torch_stable.html || exit 1

echo "✅ detectron2 설치 중..."
pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu116/torch1.13/index.html || exit 1

echo "🎉 모든 설치가 완료되었습니다!"
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda)"
python -c "import detectron2; print('Detectron2 설치 성공:', detectron2.__version__)"
