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
* chemprop>=1.5.0
* admet_ai
* pyyaml

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

POLYGON uses GPU acceleration by default for Chemprop models. Install the
CUDA-enabled versions of PyTorch and Chemprop. For example:
```
conda install cudatoolkit=11.1 -c conda-forge
pip install chemprop[gpu]
```
See <https://pytorch.org/> for platform specific installation instructions.

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

POLYGON now supports training reward function models using [Chemprop](https://github.com/chemprop/chemprop).  GPU device 0 will be used automatically when available. Provide a two-column CSV file containing `smiles` and `affinity` headers and run:

```
polygon train_reward_model \
   --training_csv my_data.csv \
   --dataset_type regression \
   --epochs 30 \
   --output_path chemprop_model.pt
```

Set `--dataset_type` to `classification` for classification tasks.

Ligand efficiency scoring can load either a pickled random forest model or a
trained Chemprop model.  The default backend is now Chemprop; specify
`model: randomforest` in your scoring definition if you prefer the RF model.

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
    file: ../data/chemprop_model_MTOR.pth
  - category: sa
    name: sa
    minimize: true
    mu: 3.0
    sigma: 1.0
  - category: bbb
    name: bbb
    minimize: false
    mu: 0.5
    sigma: 0.1
```

Use the YAML file by passing it to `--scoring_definition` when running `polygon generate`.

Additional categories include `sa` for synthetic accessibility and `bbb` for blood-brain barrier predictions using ADMET-AI.

## Example: High Affinity with Low BBB Penetration

The snippet below illustrates a typical workflow to train a target affinity model, set up a YAML scoring definition and finally generate molecules predicted to bind strongly while remaining outside the blood-brain barrier.

1. **Train a Chemprop affinity model**

   Prepare a CSV named `mtor_train.csv` with columns `smiles` and `affinity` then run:

   ```bash
   polygon train_reward_model \
       --training_csv mtor_train.csv \
       --dataset_type regression \
       --epochs 30 \
       --output_path mtor_cp.pt
   ```

2. **Create a scoring definition**

   ``scoring.yaml`` might look like:

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
       file: mtor_cp.pt
     - category: bbb
       name: bbb
       minimize: true
       mu: 0.5
       sigma: 0.1
   ```

3. **Run molecule generation**

   ```bash
   polygon generate \
       --model_path pretrained_vae_model.pt \
       --scoring_definition scoring.yaml \
       --mols_to_sample 8192 \
       --outF high_affinity_low_bbb
   ```

The resulting SMILES in `high_affinity_low_bbb.txt` are prioritized for MTOR affinity while penalizing BBB penetration.
