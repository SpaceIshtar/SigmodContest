# SigmodContest
## Members
| name | email | institution |
|------|-------|-------------|
| Jiarui Luo | 11911419@mail.sustech.edu.cn | Southern University of Science and Technology |
| Xinying Zheng | 11912039@mail.sustech.edu.cn | Southern University of Science and Technology |
| Renjie Liu || Southern University of Science and Technology |

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

## Design
1. Use regular expression to extract features from sentences
2. Find entity pairs whose features are highly matched and add them at the beginning of the result set
3. Encode each sentence using neural network
4. Build an HNSW index with the help of faiss
5. Search the index to find topk neighbors for each encoded sentence and generate (sentence, neighborhood) pairs
6. Sort the pairs using cosine distance
7. Filter the result to remove pairs that are unlikely to match using extracted features
