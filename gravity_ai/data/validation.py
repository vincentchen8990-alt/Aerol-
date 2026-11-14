"""Small validators for provenance, timestamps and appropriately grouped evaluation."""
import hashlib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def group_split(groups,test_size=.2,seed=42):
    groups=np.asarray(groups)
    if len(set(groups))<3:raise ValueError('At least 3 independent participants/sessions needed for grouped splitting')
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=seed).split(groups,groups=groups))
    assert set(groups[train]).isdisjoint(set(groups[test]))
    return train,test
