"""PhysioNet/E4 CSV reader. Aggregates aligned physical measurements into windows.

E4 formats: first row start timestamp, second row rate, subsequent rows samples.
The original source may specify UTC ISO strings or Unix seconds.
ACC raw counts are 1/64g for E4; PPG pulse rate is not cuff blood pressure.
"""
from pathlib import Path
from datetime import datetime,timezone
import csv
import numpy as np
import pandas as pd

def parse_start(value):
    try:return float(value)
    except ValueError:return pd.Timestamp(value).timestamp()

def read_e4_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f);start=next(reader);frequency=next(reader)
        t0=parse_start(start[0]);fs=float(frequency[0]);
        if fs<=0:raise ValueError('Invalid E4 frequency')
        values=[]
        for row in reader:
            if row and any(s.strip() for s in row):
                values.append([float(s) for s in row])
    arr=np.asarray(values,dtype=np.float64)
    if arr.ndim!=2 or len(arr)==0:raise ValueError('Empty signal')
    return t0,fs,arr

def session_features(directory,window_s=15.):
    """Aligned time-window features. Original session/time metadata retained."""
    directory=Path(directory)
    raw={}
    for name in ('HR','EDA','TEMP','ACC'):
        p=directory/(name+'.csv')
        if p.is_file():raw[name]=read_e4_csv(p)
    if 'HR' not in raw:raise FileNotFoundError(f'{directory}: HR.csv required for baseline demo')
    start=max(v[0] for v in raw.values())
    end=min(t0+len(arr)/fs for t0,fs,arr in raw.values())
    if end-start<window_s*3:raise ValueError('No 3+ aligned complete windows')
    windows=int((end-start)//window_s)
    rows=[]
    for i in range(windows):
        lo=start+i*window_s;hi=lo+window_s
        d={'window_start_utc_s':lo,'window_duration_s':window_s}
        for name,(t0,fs,arr) in raw.items():
            i0=max(0,int(np.ceil((lo-t0)*fs)));i1=min(len(arr),int(np.ceil((hi-t0)*fs)))
            if i1<=i0:continue
            sub=arr[i0:i1]
            if name=='ACC':
                accel=np.linalg.norm(sub/64.,axis=1)
                d['acc_norm_mean_g']=float(np.mean(accel));d['acc_norm_std_g']=float(np.std(accel))
            else:
                d[name.lower()+'_mean']=float(np.mean(sub[:,0])); d[name.lower()+'_std']=float(np.std(sub[:,0]))
        rows.append(d)
    return pd.DataFrame(rows)

def load_sessions(root):
    root=Path(root);parts=[]
    for hr in root.rglob('HR.csv'):
        s=hr.parent
        try:
            df=session_features(s)
        except (FileNotFoundError,ValueError,StopIteration):continue
        df['subject_id']=s.name.split('_')[0];df['session_id']=s.parent.name+'/'+s.name
        df['condition']=s.parent.name
        parts.append(df)
    if not parts:raise FileNotFoundError(f'No valid subject sessions under {root}; obtain real PhysioNet E4 CSVs first')
    return pd.concat(parts,ignore_index=True,sort=False)
