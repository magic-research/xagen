conda create -n xagen python=3.7
conda activate xagen
pip3 install -r requirements.txt
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip3 install "git+https://github.com/zcxu-eric/kaolin.git"