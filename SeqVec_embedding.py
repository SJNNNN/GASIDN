import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
from tqdm import tqdm
model_dir = Path('test-cache')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
embedder = ElmoEmbedder(options,weights, cuda_device=-1)
labels=[]
sequence=[]
vec_lst=[]
np.set_printoptions(threshold=np.inf)
f = open("Sub_goli/new/train_new.txt", 'r', encoding="utf-8")
lines = f.readlines()
for line in lines:
       sequence.append(line.split(' ')[0].strip())

       # labels.append(line.split(' ')[1].strip())
f.close()

num=0
#padding embedding
# for i in tqdm(sequence, desc='elmo'):
#     num += 1
#     length = len(i)
#     vec = embedder.embed_sentence(i)
#     a=np.array(vec).sum(axis=0)
#     if length >= 610:
#         length=610
#         # print(type(a[:length,:]))
#         # print(a[:length,:].shape)
#         print(num)
#         datamatrix = np.mat(a[:length, :])
#         np.save("Sub_goli/new/padding train/arr{}".format(num), datamatrix)
#     if length < 610:
#         b=np.concatenate((a[:length,:], np.zeros((610 - length, 1024))))
#         print(b.shape)
#         print(num)
#         datamatrix = np.mat(b)
#         np.save("Sub_goli/new/padding train/arr{}".format(num), datamatrix)
# no_padding embedding
for i in tqdm(sequence, desc='elmo'):
    num += 1
    length = len(i)
    vec = embedder.embed_sentence(i)
    a=np.array(vec).sum(axis=0)
    datamatrix = np.mat(a)
    print(datamatrix.shape)
    np.save("Sub_goli/new/Nopadding test/arr{}".format(num), datamatrix)

