"""Join original 2013–14 NHANES XPT files by real SEQN. No synthetic BMD samples."""
from pathlib import Path
import numpy as np
import pandas as pd

def _read(root,name):
    f=Path(root)/(name+'.XPT')
    if not f.is_file():raise FileNotFoundError(str(f))
    df=pd.read_sas(f,format='xport')
    if 'SEQN' not in df:raise ValueError(name+' missing SEQN')
    if df.SEQN.duplicated().any():raise ValueError(name+' has duplicate SEQN')
    return df

def load_nhanes_bone(root):
    # names verified against CDC 2013-14 release. Do not invent absent fields.
    demo=_read(root,'DEMO_H');bp=_read(root,'BPX_H');fem=_read(root,'DXXFEM_H');spine=_read(root,'DXXSPN_H')
    d=demo.merge(bp,on='SEQN',how='left',validate='one_to_one').merge(fem,on='SEQN',how='left',validate='one_to_one').merge(spine,on='SEQN',how='left',validate='one_to_one')
    age='RIDAGEYR';sex='RIAGENDR';pulse='BPXPLS'
    for col in (age,sex):
        if col not in d:raise ValueError('Expected CDC demographics column not present: '+col)
    # CDC DXA variables vary by file/release; do not replace them by guessed names.
    # Select reported BMD columns; retain exact original variable names for output.
    outcomes=[c for c in d.columns if c.startswith('DXX') and c.endswith('BMD')]
    if not outcomes:raise ValueError('No verified DXX*BMD outcome columns found; inspect original XPT documentation')
    systolic=[c for c in ('BPXSY1','BPXSY2','BPXSY3') if c in d]
    diastolic=[c for c in ('BPXDI1','BPXDI2','BPXDI3') if c in d]
    if systolic:d['systolic_mean_mmHg']=d[systolic].mean(axis=1)
    if diastolic:d['diastolic_mean_mmHg']=d[diastolic].mean(axis=1)
    features=[c for c in [age,sex,pulse,'systolic_mean_mmHg','diastolic_mean_mmHg','BMXBMI'] if c in d]
    return d,features,outcomes
