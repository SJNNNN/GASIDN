# GASIDN
This repository contains the source code and datasets models accompanying the our paper: GASIDN:Identification of sub-Golgi proteins with multi-scale feature fusion.
## Datasets
  Each file contains protein sequence data and corresponding labels, with each line formatted as a protein sequence followed by a space and then its label.
* The Sub-Golgi protein dataset is located in the ge folder, where train.txt serves as the training set, and test.txt serves as the test set.
* The Plant vacuole protein dataset is located in the IPVP folder, with train.txt as the training set and test.txt as the test set.
* The Peroxisomal protein dataset is stored in the PP folder, with PP_data.txt containing both the data and labels.
## Source codes
* `SeqVec_embedding.py`: Generates 1024-dimensional feature vectors to represent protein sequences using SeqVec embeddings.
* `map.py`: Calculates contact maps for protein sequences.
* `SeqVec_test_ge.py`: Trains the SAGIDN model on an independent test set of Sub-Golgi proteins.
* `SeqVeccross_ge.py`: Performs tenfold cross-validation training of the SAGIDN model.
* `model.py`:Defines the architecture of the SAGIDN model.
## Model parameter settings
* The detailed parameters of our GASIDN model can be found in the config.yaml file.
## Testable case 
We saved the model in the form of a .pkl file and uploaded it to the directory named model_storage. Additionally, we have uploaded the test code in a file called predict.py. To run the test, users only need to place the required protein embedding files and PDB files into the specified folder path, as outlined in the code. Once the files are placed correctly, the test can be executed smoothly.
```Python
python predict.py
```
# Dependencies
* python >= 3.6
* pytorch
* numpy
* pandas
* sklearn
# Contact
Due to the extraction of sequence features and structural features of Golgi, plant vesicles and peroxisomal proteins in files exceeding 12G, the files are too large to upload. If necessary, please email us.
