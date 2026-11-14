"""Optional time-series GRU/LSTM/Transformer training on actual E4 windows.

Requires enough individual subjects and a per-session continuous timeline.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from gravity_ai.models.timeseries import SequenceRegressor
from gravity_ai.data.catalog_loader import project_root

def sequence_train(csv_path,out,kind='gru',length=8,epochs=12,seed=42):
    torch.manual_seed(seed);torch.set_num_threads(2)
    df=pd.read_csv(csv_path)
    if df.subject_id.nunique()<5:raise ValueError('Requires >=5 independent real participants')
    fields=[c for c in ['hr_mean','hr_std','eda_mean','eda_std','temp_mean','temp_std','acc_norm_mean_g','acc_norm_std_g'] if c in df]
    # Sessions, not arbitrary shuffled timestamps, define the valid sequence windows.
    X=[];Y=[];groups=[]
    for _,d in df.sort_values('window_start_utc_s').groupby('session_id'):
        d=d.reset_index(drop=True)
        for k in range(length,len(d)):
            X.append(d.loc[k-length:k-1,fields].to_numpy(dtype='float32'))
            Y.append(float(d.loc[k,'hr_mean']))
            groups.append(str(d.loc[k,'subject_id']))
    X=np.asarray(X);Y=np.asarray(Y,dtype='float32');groups=np.asarray(groups)
    if len(set(groups))<5:raise ValueError('Insufficient subject groups')
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(Y,groups=groups))
    # No next-window labels from any held-out participant appear in training.
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    xscale=StandardScaler().fit(X[tr].reshape(-1,len(fields))); ymean=Y[tr].mean(); ystd=max(Y[tr].std(),1.)
    X=np.nan_to_num(xscale.transform(X.reshape(-1,len(fields))),nan=0.,posinf=0.,neginf=0.).reshape(-1,length,len(fields))
    net=SequenceRegressor(len(fields),kind=kind)
    optim=torch.optim.Adam(net.parameters(),lr=.001); train_x=torch.tensor(X[tr],dtype=torch.float32);train_y=torch.tensor((Y[tr]-ymean)/ystd,dtype=torch.float32).view(-1,1)
    for epoch in range(epochs):
        net.train();idx=torch.randperm(len(tr))
        for batch in idx.split(128):
            loss=nn.functional.mse_loss(net(train_x[batch]),train_y[batch]);optim.zero_grad();loss.backward();optim.step()
    net.eval()
    with torch.no_grad():pred=net(torch.tensor(X[te],dtype=torch.float32)).numpy().ravel()*ystd+ymean
    output={'model':kind,'data_type':'real_E4','test_mae_bpm':float(mean_absolute_error(Y[te],pred)),
            'n_train_windows':len(tr),'n_test_windows':len(te),'train_subjects':len(set(groups[tr])),'test_subjects':len(set(groups[te]))}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':net.state_dict(),'features':fields,'kind':kind,'window_length':length},out/f'physiology_{kind}.pt')
    (out/f'physiology_{kind}_metrics.json').write_text(json.dumps(output,indent=2));return output
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['gru','lstm','transformer'],default='gru');ap.add_argument('--epochs',type=int,default=12);args=ap.parse_args()
    root=project_root();p=root/'data'/'processed'/'wearable'/'e4_windows.csv'
    if not p.exists():raise SystemExit('Only real E4 participant windows can train this network; none downloaded yet')
    print(json.dumps(sequence_train(p,root/'outputs'/'checkpoints',kind=args.model,epochs=args.epochs),indent=2))
