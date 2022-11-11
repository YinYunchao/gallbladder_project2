import pandas as pd
from sklearn.metrics import roc_curve, auc,accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def confusion_cal(test_results, test_prob, test_label):
    arr = confusion_matrix(test_label,test_results)
    print('Ture Positive', arr[1,1])
    print('Ture Neg', arr[0,0])
    print('accuracy', accuracy_score(test_label,test_results))
    print('sensitivity', arr[1,1]/(arr[1,0]+arr[1,1]))
    print('specifcity', arr[0,0]/(arr[0,0]+arr[0,1]))
    FPR, TPR, Thres = roc_curve(test_label, test_prob[:, 1])
    AUC = auc(FPR, TPR)
    print('AUC', AUC)
    plt.figure()
    plt.plot(FPR, TPR,
             label='ROC curve (AUC = {0:0.2f})' ''.format(AUC),
             color='red', linewidth=4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    plt.show()


prob = pd.read_csv('X:/gallbladder_project/GnL_result/test_prediction.csv').values[:,1:]
label = pd.read_csv('X:/gallbladder_project/GnL_result/test_label.csv').values[:,1:]

pre = np.array(prob[:,0] <= prob[:,1]).astype(int)
confusion_cal(pre,prob,label)