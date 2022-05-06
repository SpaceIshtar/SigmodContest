# SigmodContest
## Members
Jiarui Luo email: 11911419@mail.sustech.edu.cn
Xinying Zheng email: 11912039@mail.sustech.edu.cn
Renjie Liu email:

## Setup
1. install anaconda3 and go into anaconda3 bash
2. create a new environment for testing: 
`conda create --name dbgroup python=3.9`
3. activate new environment: `conda activate dbgroup`
4. install basic packages: `pip install numpy pandas scipy tqdm`
5. install cpu-version pytorch: `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu`
6. install sentence-transformers: `pip install -U sentence-transformers`
7. install faiss: `conda install faiss -c pytorch`

## Run
`python Main.py`
