# GASIDN
This repository contains the source code and datasets models accompanying the our paper: GASIDN:Identification of sub-Golgi proteins with multi-scale feature fusion.
## Datasets
  Each file contains protein sequence data and labels, formatted such that each line consists of a protein sequence followed by a space and then a label.
* The sub-Golgi protein dataset is stored in the "ge" folder, with "train.txt" serving as the training set and "test.txt" as the test set.
* The Plant vacuole protein dataset is stored in the "IPVP" folder, with "train.txt" serving as the training set and "test.txt" as the test set.
* The peroxisomal protein dataset is stored in the "PP" folder, with "PP_data.txt" containing both the data and the labels.
## Source codes:
* SeqVec_embedding.py: representation of protein sequences as 1024-dimensional feature vectors.
* map.py:calculate contact maps.
* SeqVec_test_ge.py: train the SAGIDN model on the independent test set of Sub-Golgi proteins.
* SeqVeccross_ge.py: train the SAGIDN model on the tenfold cross-validation.
* model.py:the SAGIDN model.
## Experiment settings
  The model was trained using a grid search strategy to identify the optimal parameter settings for enhancing performance and training results. Various combinations of GCN hidden layer sizes (128, 256, and 512), learning rates (0.01, 0.001, and 0.0001), batch sizes (6, 8, and 10), and the number of attention heads (8, 16, 32, and 64) were explored. Additionally, the Dropout technique was implemented with values of 0.2, 0.5, and 0.7. 
  The final selected parameters were a GCN hidden layer size of 256, a learning rate of 0.0001, 16 attention heads, and a dropout rate of 0.5. The model training was conducted using Python 3.6.3 and PyTorch 1.7.1+cu101 on a single NVIDIA GTX 1080Ti GPU.
  See the file "Parameters" for other parameters.
# Dependencies
* python >= 3.6
* pytorch
* numpy
* pandas
* sklearn
# Contact
Due to the extraction of sequence features and structural features of Golgi, plant vesicles and peroxisomal proteins in files exceeding 12G, the files are too large to upload. If necessary, please email us at sjn17860639060@163.com.
