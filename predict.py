import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import torch.nn as nn
from model import Model
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score, recall_score, matthews_corrcoef
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
warnings.filterwarnings('ignore')
Dataset_Path = './ge/gej/'
Graph_Path = './ge/gej/graph/'
Result_Path = './result/'

# Seed
SEED = 64
np.random.seed(SEED)
torch.manual_seed(SEED)
def perf_measure(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
           TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
           FP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
           TN += 1
        if y_true[i] == 1 and y_pred[i] == 0:
           FN += 1
    return TP, FP, TN, FN
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)
def embedding(path):
    sequence = []
    labels = []
    features = []
    features1=[]
    f = open(path, 'r', encoding="utf-8")
    # f1 = open("test-cache/FastText_result.txt", 'w', encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        sequence.append(line.split(' ')[0])
        labels.append(line.split(' ')[1].strip())
    f.close()
    for i in range(len(labels)):
        if path == "./ge/gej/train_new.txt":
            dir = './ge/gej/SeqVec_Nopadding/train_feature/'
            print(np.load(dir + "arr" + str(i + 1) + ".npy").shape)
            features.append(np.load(dir + "arr" + str(i + 1) + ".npy"))
        if path == "./ge/gej/test_new.txt":
            dir = './ge/gej/SeqVec_Nopadding/test_feature/'
            print(np.load(dir + "arr" + str(i + 1) + ".npy").shape)
            features.append(np.load(dir + "arr" + str(i + 1) + ".npy"))
    for i in range(len(labels)):
        if path == "./ge/gej/train_new.txt":
            dir = './ge/gej/SeqVec_paddiing/train/'
            print(np.load(dir + "arr" + str(i + 1) + ".npy").shape)
            features1.append(np.load(dir + "arr" + str(i + 1) + ".npy"))
        if path == "./ge/gej/test_new.txt":
            dir = './ge/gej/SeqVec_paddiing/test/'
            print(np.load(dir + "arr" + str(i + 1) + ".npy").shape)
            features1.append(np.load(dir + "arr" + str(i + 1) + ".npy"))
    return features,labels,features1
def load_data(seq_file, graphdir):
    labels = []
    graphs = []
    num = 0
    print("Load data.")
    features,labels1,features1=embedding(seq_file)
    for i in labels1:
        labels.append(int(i))
    for i in range(len(labels)):
          graph = np.load(graphdir + "arr"+str(i+1)+ ".npy")
          graphs.append(graph)
    num += 1
    if (num % 5 == 0):
            print("load " + str(num) + " sequences")
    return features, graphs, labels,features1

pm=[]
lm=[]

def evaluate(model, val_features, val_graphs, val_labels,val_paddingfeatures):
    model.eval()

    exact_match = 0
    test_true=[]
    test_pre=[]
    p1=[]
    # print(len(val_labels))
    for i in tqdm(range(len(val_labels))):
        with torch.no_grad():
            sequence_features = torch.from_numpy(val_features[i])
            sequence_paddingfeatures= torch.from_numpy(val_paddingfeatures[i])
            sequence_graphs = torch.from_numpy(val_graphs[i])
            labels = torch.from_numpy(np.array([int(float(val_labels[i]))]))
            sequence_features = torch.squeeze(sequence_features)
            sequence_paddingfeatures = torch.squeeze(sequence_paddingfeatures)
            sequence_graphs = torch.squeeze(sequence_graphs)
            if torch.cuda.is_available():
                features = sequence_features.cuda()
                padding_features=sequence_paddingfeatures.cuda()
                graphs = sequence_graphs.cuda()
                y_true = labels.cuda()
            else:
                features = sequence_features
                padding_features = sequence_paddingfeatures
                graphs = sequence_graphs
                y_true = labels
            y_pred = model(features, graphs,padding_features)
            p = y_pred.detach().cpu().numpy().tolist()
            l = y_true.cpu().numpy().tolist()
            for j in range(len(p)):
                pm.append(p[j])
                lm.append(int(l[j]))
            p1.append(y_pred.detach().cpu().numpy().tolist()[0])
            test_pre.append(torch.max(y_pred.cpu(), 1)[1])
            test_true.append(y_true.cpu())
            if (torch.max(y_pred, 1)[1] == y_true):
                exact_match += 1
    acc = exact_match / len(val_labels)
    # print(test_true.shape)
    # print(test_pre.shape)
    F1=f1_score(test_true, test_pre)
    TP, FP, TN, FN = perf_measure(test_true,  test_pre)
    if ((TN + FP) != 0):
        Sp = TN / (TN + FP)
    else:
        Sp = 0
    Sn=recall_score(test_true, test_pre)
    # Auc=np.mean(test_auc)
    Mcc=matthews_corrcoef(test_true, test_pre)
    fpr, tpr, threshold = roc_curve(test_true, [y[1] for y in p1])  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(test_true, [y[1] for y in p1])
    area = auc(recall, precision)
    return acc, F1,Sn,Sp,Mcc,roc_auc,area

def main():
    model = Model()
    model.load_state_dict(torch.load('./model/ge_SeqVec_best_model.pkl'))
    if torch.cuda.is_available():
        model.cuda()
    test_features, test_graphs,test_labels, test_paddingfeatures = load_data(Dataset_Path + "test_new.txt",
                                                                              Graph_Path + "test_graph/")


    print(test_features)
    acc, f1,sn,sp,mcc,auc,area = evaluate(model, test_features, test_graphs, test_labels, test_paddingfeatures)
    print("test_acc:", acc,"test_f1-score:",f1,"test_sn:",sn,"test_sp:",sp,"test_mcc:",mcc,"test_auc:",auc,"test_PRauc:",area)

if __name__ == "__main__":
    main()