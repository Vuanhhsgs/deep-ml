import numpy as np

def accuracy_score(y_true, y_pred):
    n=len(y_pred)
    ans=0
    for i in range(n):
        if(y_true[i]==y_pred[i]):
            ans+=1
    return (ans/n)