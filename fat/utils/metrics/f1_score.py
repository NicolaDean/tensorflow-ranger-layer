
def recompute_f1(TP,FP,FN):
        if (TP + FP) != 0:
            precision = TP / (TP + FP) 
        else:
            precision = 0

        if (TP + FN) != 0:
            recall    = TP / (TP + FN)
        else:
            recall = 0

        if (precision + recall) != 0:
            f1_score  = (2*precision*recall)/(precision + recall)
        else:
            f1_score = None

        if (FP+FN+TP) != 0:
            accuracy_score = (TP) / (FP+FN+TP)
        else:
            accuracy_score = None

        return precision,recall,f1_score,accuracy_score