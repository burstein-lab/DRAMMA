# DRAMMA: Detection of Resistance to AntiMicrobials using Machine-learning Approaches

## Summary

Antibiotics are essential for medical procedures, food security, and public health. However, ill-advised usage leads to increased pathogen resistance to antimicrobial substances, posing a threat of fatal infections and limiting the benefits of antibiotics. Therefore, early detection of antimicrobial resistance genes (ARGs), especially in human pathogens, is crucial.

DRAMMA (Detection of Resistance to AntiMicrobials  using Machine-learning Approaches) is a multifaceted machine-learning approach for novel antimicrobial resistance gene detection in metagenomic data. Unlike most existing methods that rely on sequence similarity to a predefined gene database, DRAMMA can predict new ARGs with no sequence similarity to known resistance or even annotated genes.

## Features

- Utilizes various features including protein properties, genomic context, and evolutionary patterns
- Demonstrates robust performance in cross-validation and external validation
- Enables rapid ARG identification for large-scale genetic and metagenomic samples
- Potential for early detection of specific ARGs, influencing antibiotic selection for patients

The trained models and data needed for the feature extraction scripts are available on Zenodo: [https://doi.org/10.5281/zenodo.12621924](https://doi.org/10.5281/zenodo.12621924)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/DRAMMA.git
   cd DRAMMA
   ```

2. Create a conda environment (optional but recommended):
   ```
   conda env create -f environment.yml
   conda activate DRAMMA
   ```

### Requirements

### Python and Libraries
- Python 3.10+
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- seaborn
- Bio
- mpmath

### External Programs
- HMMER version 3.2.1
- TMHMM 2.0c
- MMseqs2

Please ensure these external programs are installed and accessible in your system's PATH, or provide their full paths when running the scripts.

## Usage

The main scripts for DRAMMA are:

### 1. run_model.py

This script runs the trained model on input data.

```
python run_model.py -pkl <path_to_pickle> -in <input_file> -out <output_file> [options]

Options:
  -pkl, --pickle        Path to pickle with the model, relevant cols, and model score thresholds dictionary (created by create_model_pkl.py or downloaded from Zenodo)
  -in, --input_file     Path to the file we want to run the model against
  -out, --output_file   Path to pkl file we want to save our results in
  -fp, --filter_pos     Choose this to keep only negative proteins (non-AMRs) (default)
  -kp, --keep_pos       Choose this to keep both positive (known AMRs) and negative proteins (non-AMRs)
  -sc, --single_class   Choose this to run a binary model (default)
  -mc, --multi_class    Choose this to run a multi_class model
```

### 2. create_model_pkl.py

This script creates a pickle file with all the relevant information for running the model (model, relevant cols, and model score thresholds dictionary).

```
python create_model_pkl.py -i <input_path> -o <output_path> [options]

Options:
  -i, --input_path      Path to input dataset
  -o, --output_path     Path to output pickle
  -nj, --n_jobs         Number of jobs to use to train the model (default: 2)
  -ts, --test_set       Path to the test set for model evaluation (optional)
  --param_pkl           Path to pickle with chosen model hyperparameters (optional)
  --feature_json        Path to JSON file with the chosen features. (optional, default is the parameters used to train DRAMMA. Choose '' for the default Random forest hyperparameters)
  -n                    Number of top features to choose (default: 30, choose 0 to use all not correlated features)
  -mc, --is_multiclass  Create a model for the multiclass classifier (default: False)
  -lc, --label_col      Column to take for labeling (default: 'Updated Resistance Mechanism')
```

### 3. run_features.py

This script extracts features from input data. The script assumes there are four files for each assembly - protein fasta, genes, gff file, and contig file (faa, ffn, gff, fa).

```
python run_features.py --input_path <input_path> --data_path <data_path> --hmmer_path <path_to_hmmer> --mmseqs_path <path_to_mmseqs> --tmhmm_path <path_to_tmhmm> [options]

Options:
  --input_path          Full path of the directory with all the assemblies
  --data_path           Full path of the directory with all the data needed for feature extraction (feature_extraction directory downloaded from Zenodo)
  --hmmer_path          Full path to the HMMER's hmmsearch program
  --mmseqs_path         Full path to the Mmseqs2 program
  --tmhmm_path          Full path to the tmhmm program
  -k, --kmer            Run kmers count from 2 to k (default: 4)
  -lt, --label_threshold Threshold for the proximity feature (default: '1e-10')
  -t, --threshold_list  List of thresholds for proximity feature (default: [1e-8])
  -d, --gene_window     Size of the ORFs window (default: 10)
  -n, --nucleotide_window Size of the nucleotides window (default: 10000)
  -e, --by_evalue       Use threshold by e-value (default)
  -s, --by_score        Use threshold by score
  --dif_format_paths    Paths to data in different formats (faa, fa, gff, ffn) (optional, use only if you want to extract the features on a single assembly.)
  -ftd, --features_to_drop List of features to exclude (default: ['Cross_Membrane'])
  -pkl, --pickle_file   Path to pickle file with a FeatureList object (optional, if its not supplied, a new object will be created)
```

### 4. train_dataset_creator.py
This script creates a dataset for training the model, either a balanced subset of the proteins or the complete dataset.
```
python train_dataset_creator.py -d <directory> -f <fasta_file> [options]
Options:
-d, --directory       Directory containing all the feature pkl files created by run_features.py
-f, --fasta           Path to relevant protein fasta file for de-duplication. only used when all_data is false.
-wl, --whitelist      Filter which folders to check (default: '' - checks all files and directories in --directory)
-p, --pumps           Create the pump train set (default: False)
-ad, --all_data       Create dataset on entire data instead of balanced set (default: False)
-pkl, --pickle        Save data to pickle instead of tsv (default: True)
-b, --batch_size      Batch size for saving dataset when all_data=True (default: 0 - everything in one file)
-c, --columns         JSON file with the columns to include in the dataset (default: '' - use all columns)
```

## License

This project is licensed under the MIT License. This license is appropriate for academic work as it allows for free use, modification, and distribution while still providing some protection and attribution to the original authors.

```
MIT License

Copyright (c) 2024 [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact
For questions about DRAMMA, please contact us:
Ella Rannon: [ellarannon@mail.tau.ac.il](mailto:ellarannon@mail.tau.ac.il)
David Burstein: [davidbur@tauex.tau.ac.il](mailto:davidbur@tauex.tau.ac.il)
<!--

## Citation

If you use DRAMMA in your research, please cite our paper:

(Add citation information here when available)

-->
