import os
import torch
from sklearn.metrics import precision_recall_curve
import numpy as np
from tqdm import tqdm
import warnings
from model import Model
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score, recall_score, matthews_corrcoef
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.model_selection import KFold
KF = KFold(n_splits=10,shuffle=True,random_state=150)
warnings.filterwarnings('ignore')
Dataset_Path = './ge/gej/'
Graph_Path = './ge/gej/graph/'
Result_Path = './result/'
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
from tqdm import tqdm
model_dir = Path('test-cache')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
embedder = ElmoEmbedder(options,weights, cuda_device=-1)
path= "./ge/gej/"
dataset="train_new.txt"
idx_features_labels = np.genfromtxt("{}{}".format(path, dataset),
                               dtype=np.dtype(str))
X_train=idx_features_labels[0:284, 0:-1]
vec_lst=[]
np.set_printoptions(threshold=np.inf)
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
def embedding(path):
    sequence = []
    labels = []
    features = []
    features1 = []
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
    return features, labels, features1
# Seed
SEED = 2333
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(3)
    torch.cuda.manual_seed(SEED)
# Model parameters
NUMBER_EPOCHS = 100
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def evaluate(model, val_features, val_graphs, val_labels,val_paddingfeatures,test_index):
    model.eval()
    epoch_loss_valid = 0.0
    exact_match = 0
    test_true=[]
    test_pre=[]
    p1=[]
    # print(len(val_labels))
    for i in test_index:
        with torch.no_grad():
            sequence_features = torch.from_numpy(val_features[i])
            sequence_paddingfeatures= torch.from_numpy(val_paddingfeatures[i])
            sequence_graphs = torch.from_numpy(val_graphs[i])
            labels = torch.from_numpy(np.array([float(val_labels[i])]))
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
            loss = model.criterion(y_pred, y_true.long())
            epoch_loss_valid += loss.item()
    epoch_loss_valid_avg = epoch_loss_valid / len(test_index)
    acc = exact_match / len(test_index)
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
    return acc, epoch_loss_valid_avg,F1,Sn,Sp,Mcc,roc_auc,area
def train(model, epoch):
    train_features, train_graphs, train_labels, train_paddingfeatures = load_data(Dataset_Path + "train_new.txt",
                                                                                  Graph_Path + "train_graph/")
    # val_features, val_graphs, val_labels, val_paddingfeatures = load_data(Dataset_Path + "test.txt",
    #                                                                       Graph_Path + "test_contact_map/")
    ACC1=[]
    F11=[]
    Sp1=[]
    Sn1=[]
    MCC1=[]
    Auc1=[]
    Prauc=[]
    for train_index, test_index in KF.split(X_train):
        Acc=[]
        Mcc=[]
        F1=[]
        Sp=[]
        Sn=[]
        Auc=[]
        prauc=[]
        best_acc = 0
        best_epoch = 0
        cur_epoch = 0
        exact_match = 0
        # print("epoch:" + str(0))
        # print("========== Evaluate Valid set ==========")
        # valid_acc, epoch_loss_valid_avg,f1,sn,sp,mcc,auc = evaluate(model, val_features, val_graphs, val_labels)
        # print("valid acc:", valid_acc,"valid f1-score:",f1,"valid sn:",sn,"valid sp:",sp,"valid mcc:",mcc,"valid auc:",auc)
        # print("valid loss:", epoch_loss_valid_avg)
        # best_acc = valid_acc
        # best_f1 = f1
        # best_sn = sn
        # best_sp= sp
        for epoch in range(epoch):
            model.train()
            for j,i in enumerate(train_index):
                sequence_features = torch.from_numpy(train_features[i])
                sequence_paddingfeatures = torch.from_numpy(train_paddingfeatures[i])
                sequence_graphs = torch.from_numpy(train_graphs[i])
                labels = torch.from_numpy(np.array([float(train_labels[i])]))
                sequence_features = torch.squeeze(sequence_features)
                sequence_paddingfeatures = torch.squeeze(sequence_paddingfeatures)
                sequence_graphs = torch.squeeze(sequence_graphs)
                if torch.cuda.is_available():
                    features = sequence_features.cuda()
                    padding_features = sequence_paddingfeatures.cuda()
                    graphs = sequence_graphs.cuda()
                    y_true = labels.cuda()
                else:
                    features = sequence_features
                    padding_features = sequence_paddingfeatures
                    graphs = sequence_graphs
                    y_true = labels
                y_pred = model(features, graphs, padding_features)
                loss = model.criterion(y_pred, y_true.long())
                loss /= BATCH_SIZE
                loss.backward()

                if (j % BATCH_SIZE == 0):
                    model.optimizer.step()
                    model.optimizer.zero_grad()
                if (torch.max(y_pred, 1)[1] == y_true):
                    exact_match += 1
            acc = exact_match / len(train_labels)
            # print(acc)
            print("epoch:" + str(epoch + 1))
            print("========== Evaluate Valid set ==========")
            valid_acc, epoch_loss_valid_avg, f1, sn, sp, mcc, auc,area = evaluate(model, train_features, train_graphs, train_labels, train_paddingfeatures,test_index)
            Acc.append(valid_acc)
            F1.append(f1)
            Sn.append(sn)
            Sp.append(sp)
            Mcc.append(mcc)
            Auc.append(auc)
            prauc.append(area)
            # print("valid acc:", valid_acc, "valid f1-score:", f1, "valid sn:", sn, "valid sp:", sp, "valid mcc:", mcc,
            #           "valid auc:", auc)
            # print("valid loss:", epoch_loss_valid_avg)
            if best_acc < valid_acc:
                    best_acc = valid_acc
                    best_epoch = epoch + 1
                    cur_epoch = 0
                    torch.save(model.state_dict(), os.path.join('./model/ge_cross_best_model.pkl'))
            else:
                    cur_epoch += 1
                    if (cur_epoch > 200):
                        break
        # print("Best epoch at", str(best_epoch))
        # print("Best acc at", str(best_acc))
        ACC1.append(np.mean(Acc))
        F11.append(np.mean(F1))
        Sn1.append(np.mean(Sn))
        Sp1.append(np.mean(Sp))
        MCC1.append(np.mean(Mcc))
        Auc1.append(np.mean(Auc))
        Prauc.append(np.mean(prauc))
        print("valid acc:", np.mean(Acc), "valid f1-score:",np.mean(F1) , "valid sn:", np.mean(Sn), "valid sp:", np.mean(Sp), "valid mcc:", np.mean(Mcc),
                      "valid auc:", np.mean(Auc), "valid PRauc:", np.mean(prauc))
    print("valid acc:", np.mean(ACC1), "valid f1-score:", np.mean(F11), "valid sn:", np.mean(Sn1), "valid sp:",
          np.mean(Sp1), "valid mcc:", np.mean(MCC1),
          "valid auc:", np.mean(Auc1), "valid PRauc:", np.mean(Prauc))
def main():
    model = Model()
    # model.load_state_dict(torch.load('./model/best_model.pkl'))
    if torch.cuda.is_available():
        model.cuda()
    train(model, NUMBER_EPOCHS)
    f = open("./result/ge_cross_Roc_result.txt", 'w', encoding="utf-8")
    for j in range(len(pm)):
        f.writelines(str(pm[j][1]) + " " + str(int(lm[j])) + "\n")
    f.close()


if __name__ == "__main__":
    main()
