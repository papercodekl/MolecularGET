# ICLR2020_MGET

This is the primary code for the "Molecular Graph Enhanced Transformer for Retrosynthesis  Prediction".

## Install requirements

Create a new conda environment:

```bash
conda create -n mget python=3.7
source activate mget
conda install rdkit -c rdkit
conda install future six tqdm pandas
```

The code was tested for pytorch 0.4.1, to install it go on [Pytorch](https://pytorch.org/get-started/locally/).
Select the right operating system and CUDA version and run the command, e.g.:

```bash
conda install pytorch=0.4.1 torchvision -c pytorch
```
Then,
```bash
pip install torchtext==0.3.1
pip install -e . 
```
Then, install DGL
```
pip install dgl
```


Besides, you have to replace three source files(batch.py, field.py, iterator.py) of the torchtext library in "torchtext/data"  with the corrsponding three files contained in "replace_torchtext" since we have modified some codes in these files.


## Preprocessing

```
bash pre.sh
```

## Train the model
The "data2" contains UPSTO-50K without reaction type. To train the model, 
```
bash train.sh
```
The parameter settings we used can be found in "train.sh". You can modify the saving location of the model (default is experiments/checkpoints2). 

## Translation
To generate the output SMILES, 
```
bash trans.sh
```
Default settings is to generate top-10 candidates.

## Evaluation

To evaluate our model, 
```
bash eval.sh
```
