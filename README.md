# SAGIDN
This repository contains the source code and datasets models accompanying the our paper: Identification of sub-Golgi protein localization by use of multi-dimensional feature fusion.
## Source codes:
* SeqVec_embedding.py: representation of protein sequences as 1024-dimensional feature vectors.
* map.py:calculate contact maps.
* SeqVec_test_ge.py: train the SAGIDN model on the independent test set of Sub-Golgi proteins.
* SeqVeccross_ge.py: train the SAGIDN model on the tenfold cross-validation.
* model.py:the SAGIDN model.
# Dependencies
* python >= 3.6
* pytorch
* numpy
* pandas
* sklearn

# Contact
Due to the extraction of sequence features and structural features of Golgi, plant vesicles and peroxisomal proteins in files exceeding 12G, the files are too large to upload. If necessary, please email us at sjn17860639060@163.com.
