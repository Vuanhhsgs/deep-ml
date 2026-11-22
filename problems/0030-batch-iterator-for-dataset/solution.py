import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    ans = []
    sample_size = X.shape[0]


    num_batch = (sample_size-1)//batch_size + 1
    if y is not None:
        for i in range(0, batch_size * (num_batch-1), batch_size):
            X_batch = []
            y_batch = []
            for j in range(batch_size):
                X_batch.append(X[i+j])
                y_batch.append(y[i+j])
            this_batch = []
            this_batch.append(X_batch)
            this_batch.append(y_batch)
            ans.append(this_batch)

        last_batch = []
        last_X_batch = []
        last_y_batch = []
        for i in range(batch_size * (num_batch-1), sample_size):
            last_X_batch.append(X[i])
            last_y_batch.append(y[i])
        last_batch.append(last_X_batch)
        last_batch.append(last_y_batch)
        ans.append(last_batch)

    else:
        for i in range(0, batch_size * (num_batch-1), batch_size):
            this_batch = []
            for j in range(batch_size):
                this_batch.append(X[i+j])
            ans.append(this_batch)
        last_batch = []
        for i in range(batch_size * (num_batch-1), sample_size):
            last_batch.append(X[i])
        ans.append(last_batch)

    return ans
