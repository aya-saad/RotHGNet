import torch
from sklearn import metrics

def METRIX(y_true: torch.Tensor, y_pred: torch.Tensor,is_training=False) -> torch.Tensor:
    '''Calculate Accuracy, Recall, Precision and F1 score. Can work with gpu tensors
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
    '''
    print('y_true', y_true)
    print('y_pred', y_pred)
    print('confusion matrix:\n ',metrics.confusion_matrix(y_true, y_pred))
    # Print the precision and recall, among other metrics
    print('classification report:\n ', metrics.classification_report(y_true, y_pred, digits=3))
    #print(metrics.multilabel_confusion_matrix(y_true, y_pred))
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print('accuracy: ', metrics.accuracy_score(y_true, y_pred))
    balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
    print('balanced accuracy: ', metrics.balanced_accuracy_score(y_true, y_pred))
    precision = metrics.precision_score(y_true,y_pred, average='weighted')
    print('precision: ', metrics.precision_score(y_true,y_pred, average='weighted'))
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    print('recall: ', metrics.recall_score(y_true, y_pred, average='weighted'))
    f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
    print('f1 score', f1_score)
    '''
    y_true = y_true.to("cpu")
    y_pred = y_pred.to("cpu")
    return metrics.accuracy_score(y_true, y_pred),\
           metrics.balanced_accuracy_score(y_true, y_pred),\
           metrics.precision_score(y_true,y_pred, average='weighted'),\
           metrics.recall_score(y_true, y_pred, average='weighted'),\
           metrics.f1_score(y_true, y_pred, average='weighted'),\
           metrics.classification_report(y_true, y_pred, digits=3)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fake NN output
    #out = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9], [0.9, 0.05, 0.05]])

    out = torch.FloatTensor([[0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711],
                             [0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711],
                             [0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711],
                             [0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711],
                             [0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711],
                             [0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711],
                             [0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711],
                             [0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711],
                             [0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711],
                             [0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711],
                             [0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711],
                             [0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711],
                             [0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711],
                             [0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711],
                             [0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711],
                             [0.1649, 0.1626, 0.1540, 0.1644, 0.1830, 0.1711]])
    #out = torch.FloatTensor([[0.1612, 0.1684, 0.1534, 0.1591, 0.1806, 0.1774],
    #                         [0.1612, 0.1684, 0.1534, 0.1591, 0.1806, 0.1774],
    #                         [0.1612, 0.1684, 0.1534, 0.1591, 0.1806, 0.1774],
    #                         [0.1612, 0.1684, 0.1534, 0.1591, 0.1806, 0.1774]])
    out = torch.autograd.Variable(out)

    # Categorical targets
    #tensor([3, 3, 3, 4, 1, 5, 0, 0, 2, 0, 3, 1, 2, 2, 1, 0])

    y = torch.LongTensor([[3],[3],[3],[4],[1],[5],[0],[0],[2],[0],[3],[1],[2],[2],[1],[0]])
    #y = torch.LongTensor([[3], [0], [2], [2]])
    y = y.to("cpu")
    # ([[0.1806],[0.1806],
    #         [0.1806],
    #         [0.1806]])
    #y = torch.LongTensor([1, 2, 0])
    y = torch.autograd.Variable(y)
    accuracy, balanced_accuracy, precision, recall, f1_score, rep = METRIX(y.squeeze(1).long(), out)

    print('Acc: {:.3f}%, Balanced Acc.: {:.3f}%, Precision: {:.3f}%, '
          'Recall: {:.3f}%, F1 Score: {:.3f}%'.format(accuracy*100, balanced_accuracy*100,precision*100, recall*100, f1_score*100))
    print('Report: \n', rep)

'''
    # the following calculations of the TP and TN are based on a binary representation 
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    print('tp: ', tp, ' tn: ', tn, ' fp: ', fp, ' fn: ', fn)

    epsilon = 1e-7
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
'''