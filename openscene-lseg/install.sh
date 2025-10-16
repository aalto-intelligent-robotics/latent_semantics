# before running this, run:
# conda env create -n lseg python=3.8
# conda activate lseg
pip install -U pip

pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# install clip
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

#pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/@331ecdd5306104614cb414b16fbcd9d1a8d40e1e  # this step takes >5 minutes
pip install torch-encoding


pip install timm==0.5.4
pip install torchmetrics==0.6.0
pip install setuptools==59.5.0
pip install imageio matplotlib pandas six

#ADDS
pip install Ninja
pip install pytorch-lightning==1.3.5

pip install filelock
pip install transformers==4.28.0
