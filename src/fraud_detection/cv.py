import numpy as np

class TimeAwareStratifiedSplit:
    """
    Expanding-window CV splitter for imbalanced, time-ordered data.

    Splits the (already time-ordered) training data into `n_splits + 1` sequential
    blocks. Fold k trains on everything up to block k and validates ONLY on block
    k+1 — the model never trains on data that is temporally after its validation
    set, unlike random StratifiedKFold.

    Because fraud is rare (~3.5%), a validation block can end up with very few or
    zero positive cases. If a block has fewer than `min_pos` fraud rows, it's
    merged forward into the next block until the minimum is met, so every fold
    stays statistically usable.
    """
    def __init__(self, n_splits: int = 5, min_pos: int = 30):
        self.n_splits = n_splits
        self.min_pos = min_pos

    def split(self, X, y, time_values):
        order = np.argsort(np.asarray(time_values))
        y_arr = np.asarray(y)
        n = len(order)
        block_size = n // (self.n_splits + 1)

        blocks = [order[i * block_size:(i + 1) * block_size] for i in range(self.n_splits + 1)]
        leftover = order[(self.n_splits + 1) * block_size:]
        if len(leftover):
            blocks[-1] = np.concatenate([blocks[-1], leftover])

        folds = []
        train_idx = list(blocks[0])
        i = 1
        while i <= self.n_splits:
            val_idx = list(blocks[i])
            while y_arr[val_idx].sum() < self.min_pos and i < self.n_splits:
                i += 1
                val_idx += list(blocks[i])
            folds.append((list(train_idx), val_idx))
            train_idx += val_idx
            i += 1
        if len(folds) >= 2 and y_arr[folds[-1][1]].sum() < self.min_pos:
            prev_train, prev_val = folds[-2]
            _, last_val = folds[-1]
            folds[-2] = (prev_train, prev_val + last_val)
            folds.pop()

        for train_idx, val_idx in folds:
            yield np.array(train_idx), np.array(val_idx)