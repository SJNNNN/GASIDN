import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from model import Model
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score, recall_score, matthews_corrcoef
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
warnings.filterwarnings('ignore')
Dataset_Path = './IPVP/'
Graph_Path = './IPVP/graph/'
Result_Path = './result/'
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
from tqdm import tqdm
model_dir = Path('test-cache')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
embedder = ElmoEmbedder(options,weights, cuda_device=-1)
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
    features1=[]
    f = open(path, 'r', encoding="utf-8")
    # f1 = open("test-cache/FastText_result.txt", 'w', encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        sequence.append(line.split(' ')[0])
        labels.append(line.split(' ')[1].strip())
    f.close()
    for i in range(len(labels)):
        if path == "./IPVP/train.txt":
            dir = './IPVP/SeqVecfeature_nopadding/train/'
            print(np.load(dir + "arr" + str(i + 1) + ".npy").shape)
            features.append(np.load(dir + "arr" + str(i + 1) + ".npy"))
        if path == "./IPVP/test.txt":
            dir = './IPVP/SeqVecfeature_nopadding/test/'
            print(np.load(dir + "arr" + str(i + 1) + ".npy").shape)
            features.append(np.load(dir + "arr" + str(i + 1) + ".npy"))
    for i in range(len(labels)):
        if path == "./IPVP/train.txt":
            dir = './IPVP/SeqVec_feature/train/'
            print(np.load(dir + "arr" + str(i + 1) + ".npy").shape)
            features1.append(np.load(dir + "arr" + str(i + 1) + ".npy"))
        if path == "./IPVP/test.txt":
            dir = './IPVP/SeqVec_feature/test/'
            print(np.load(dir + "arr" + str(i + 1) + ".npy").shape)
            features1.append(np.load(dir + "arr" + str(i + 1) + ".npy"))
    return features,labels,features1
# Seed
SEED = 2333
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(3)
    torch.cuda.manual_seed(SEED)
# Model parameters
NUMBER_EPOCHS = 100
BATCH_SIZE = 10
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

def evaluate(model, val_features, val_graphs, val_labels,val_paddingfeatures):
    model.eval()
    epoch_loss_valid = 0.0
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
            p = y_pred.detach().numpy().tolist()
            l = y_true.numpy().tolist()
            for j in range(len(p)):
                pm.append(p[j])
                lm.append(int(l[j]))
            p1.append(y_pred.detach().numpy().tolist()[0])
            test_pre.append(torch.max(y_pred, 1)[1])
            test_true.append(y_true)
            if (torch.max(y_pred, 1)[1] == y_true):
                exact_match += 1
            loss = model.criterion(y_pred, y_true.long())
            epoch_loss_valid += loss.item()
    epoch_loss_valid_avg = epoch_loss_valid / len(val_labels)
    acc = exact_match / len(val_labels)
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
    train_features, train_graphs, train_labels ,train_paddingfeatures= load_data(Dataset_Path + "train.txt",
                                                           Graph_Path + "train_contact_map/")
    val_features, val_graphs, val_labels ,val_paddingfeatures= load_data(Dataset_Path + "test.txt",
                                                           Graph_Path + "test_contact_map/")

    best_acc = 0
    best_epoch = 0
    cur_epoch = 0
    exact_match = 0
    print("epoch:" + str(0))
    print("========== Evaluate Valid set ==========")
    valid_acc, epoch_loss_valid_avg,f1,sn,sp,mcc,auc,area = evaluate(model, val_features, val_graphs, val_labels,val_paddingfeatures)
    print("valid acc:", valid_acc,"valid f1-score:",f1,"valid sn:",sn,"valid sp:",sp,"valid mcc:",mcc,"valid auc:",auc,"valid PRauc:",area)
    print("valid loss:", epoch_loss_valid_avg)
    best_acc = valid_acc
    best_f1 = f1
    best_sn = sn
    best_sp= sp
    for epoch in range(epoch):
        model.train()
        for i in tqdm(range(len(train_labels))):
            sequence_features = torch.from_numpy(train_features[i])
            sequence_paddingfeatures = torch.from_numpy(train_paddingfeatures[i])
            sequence_graphs = torch.from_numpy(train_graphs[i])
            labels = torch.from_numpy(np.array([int(float(train_labels[i]))]))
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
            loss = model.criterion( y_pred , y_true.long())
            loss /= BATCH_SIZE
            loss.backward()

            if (i % BATCH_SIZE == 0):
                model.optimizer.step()
                model.optimizer.zero_grad()
            if (torch.max(y_pred, 1)[1] == y_true):
                     exact_match += 1
        acc = exact_match / len(train_labels)
        # print(acc)
        print("epoch:" + str(epoch + 1))
        print("========== Evaluate Valid set ==========")
        valid_acc, epoch_loss_valid_avg, f1, sn, sp, mcc, auc,area = evaluate(model, val_features, val_graphs, val_labels,val_paddingfeatures)
        print("valid acc:", valid_acc, "valid f1-score:", f1, "valid sn:", sn, "valid sp:", sp, "valid mcc:", mcc,
              "valid auc:", auc, "valid PRauc:", area)
        print("valid loss:", epoch_loss_valid_avg)
        if best_acc < valid_acc:
            best_acc = valid_acc
            best_epoch = epoch + 1
            cur_epoch = 0
            torch.save(model.state_dict(), os.path.join('./model/SeqVec_best_model_noattention.pkl'))
        else:
            cur_epoch += 1
            if (cur_epoch > 200):
                break
    print("Best epoch at", str(best_epoch))
    print("Best acc at", str(best_acc))
def main():
    model = Model()
    # model.load_state_dict(torch.load('./model/best_model.pkl'))
    if torch.cuda.is_available():
        model.cuda()
    train(model, NUMBER_EPOCHS)
    f = open("./result/yepao_independ_Roc_result_noattention.txt", 'w', encoding="utf-8")
    for j in range(len(pm)):
        f.writelines(str(pm[j][1]) + " " + str(int(lm[j])) + "\n")
    f.close()
    fpr, tpr, threshold = roc_curve(lm, [y[1] for y in pm])  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    lw = 2
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.6f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()