# Generative Models do not Go Mad
Code to reproduce the experiments of the ICLR 2024 (spotlight) paper "On the Stability of Iterative Retraining of Generative Models on their Own Data" https://arxiv.org/abs/2310.00429. You will need to install the following packages: edm, fls, conditional-flow-matching (and optionally ddpm-torch)

Install

git clone https://github.com/NVlabs/edm.git

cd edm

conda env create -f environment.yml -n edm

conda activate edm

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

cd ..

git clone https://github.com/marcojira/fls.git

cd fls

pip install -e .

pip install wandb

cd ..

git clone git@github.com:atong01/conditional-flow-matching.git

cd conditional-flow-matching

pip install -e .
