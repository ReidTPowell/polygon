# POLYpharmacology Generative Optimization Network (POLYGON) a VAE for de novo polypharmacology.

This repository contains the POLYGON framework, a de novo molecular generator for polypharmacology. Akin to de novo portait generation, POLYGON attempts to optimize the chemical space for multiple protein target domains.

![alt text](https://github.com/bpmunson/polygon/blob/main/images/Figure_1r.png?raw=true)

***

The codebase is primarily adapted from two excellent de novo molecular design frameworks:

1. GuacaMol for reward based reinforcement learning: https://github.com/BenevolentAI/guacamol 

2. MOSES for the VAE implementation: https://github.com/molecularsets/moses

## Data Sources
A key resource to the POLYGON framework is experimental binding data of small molecule ligands.  We use the BindingDB as a source for this information, which can be found here: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp

Input molecule training datasets are available from the GuacaMol package:  https://github.com/BenevolentAI/guacamol 

## Installation of POLYGON:
POLYGON has been testing on Python version 3.9.16.

Installation of POLYGON with pip will automatically install the necessary dependencies, which are:
* pandas>=1.0.3
* numpy>=1.18.1
* rdkit>=2019.09.3
* torch>=1.4.0
* joblib>=0.14.1
* scikit-learn>=0.22.1

```
conda install -c conda-force rdkit
conda install pytorch::pytorch -c pytorch
conda install numpy pandas scikit-learn

```

```
git clone https://github.com/bpmunson/polygon.git

cd polygon

pip install .
```

optionally install cudatoolkit for gpu acceleration in pytorch
for example:
```
conda install cudatoolkit=11.1 -c conda-forge
```
or see https://pytorch.org/ for specific installation instructions.

Installation time is on the order of minutes.

***


Example Usage:

Pretrain VAE to encode chemical embedding:
```
polygon train \
	--train_data ../data/guacamol_v1_train.smiles \
	--log_file log.txt \
	--save_frequency 25 \
	--model_save model.pt \
	--n_epoch 200 \
	--n_batch 1024 \
	--debug \
        --d_dropout 0.2 \
        --device cpu
```

The training command now streams SMILES directly from disk using a lazy dataset,
which allows handling files that do not fit into RAM.

Train Ligand Binding Models for Two Protein Targets
```
polygon train_ligand_binding_model \
   --uniprot_id Q02750
   --binding_db_path BindingDB_All.csv
   --output_path Q02750_ligand_binding.pkl
```

```
polygon train_ligand_binding_model \
   --uniprot_id P42345
   --binding_db_path BindingDB_All.csv
   --output_path P42345_ligand_binding.pkl
```

Use the chemical embedding to design polypharmacology compounds
```
polygon generate \
    --model_path ../data/pretrained_vae_model.pt \
    --scoring_definition scoring_definition.csv \
    --max_len 100 \
    --n_epochs 200 \
    --mols_to_sample 8192   \
    --optimize_batch_size 512    \
    --optimize_n_epochs 2   \
    --keep_top 4096   \
    --opti gauss   \
    --outF molecular_generation   \
    --device cpu  \
    --save_payloads   \
    --n_jobs 4 \
    --debug
```

The expected runtime for POLYGON is on the order of hours.

POLYGON will output designs as SMILES strings in a text file.  For example:
```
$ head GDM_final_molecules.txt
Fc1cc(F)cc(CC(Nc2ccc3ncccc3c2)c2cccnc2)c1
N[SH](=O)(O)c1cccc(S(=O)(=O)O)c1
N#Cc1cc(C(N)=NO)ccc1Nc1nccc2ccnn12
CN(CN=C(O)c1ccco1)Nc1nccs1
```

## Training Reward Models with Chemprop

POLYGON now supports training reward function models using [Chemprop](https://github.com/chemprop/chemprop).  Provide a two-column CSV file containing `smiles` and `affinity` headers and run:

```
polygon train_reward_model \
   --training_csv my_data.csv \
   --dataset_type regression \
   --epochs 30 \
   --output_path chemprop_model.pt
```

Set `--dataset_type` to `classification` for classification tasks.

## YAML Scoring Definitions

Scoring definitions used during molecule generation can now be provided as YAML in addition to CSV.  A YAML file should contain a list under the key `scoring` with the same fields as the original CSV, for example:

```yaml
scoring:
  - category: qed
    name: qed
    minimize: false
    mu: 0.67
    sigma: 0.1
  - category: ligand_efficiency
    name: MTOR_le
    minimize: false
    mu: 0.8
    sigma: 0.3
    file: ../data/RFR_ligand_binding_model_MTOR.pkl
```

Use the YAML file by passing it to `--scoring_definition` when running `polygon generate`.
