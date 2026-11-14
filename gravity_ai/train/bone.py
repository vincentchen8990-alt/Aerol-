"""NHANES cross-sectional BMD prediction only (not gravity dose response)."""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from gravity_ai.data.catalog_loader import project_root
from gravity_ai.models.bone import models
import joblib

def train(data_path,outdir):
    df=pd.read_csv(data_path)
    if 'SEQN' not in df or df.SEQN.duplicated().any():raise ValueError('SEQN must be unique')
    outcomes=[c for c in df if c.startswith('DXX') and c.endswith('BMD')]
    if not outcomes:raise ValueError('No genuine DXA BMD outcome fields')
    features=[c for c in ['RIDAGEYR','RIAGENDR','BPXPLS','systolic_mean_mmHg','diastolic_mean_mmHg'] if c in df]
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    results={'data_type':'real_NHANES_cross_sectional','not_longitudinal':True,'outcomes':{}}
    for target in outcomes[:5]:
        dat=df.dropna(subset=[target]).copy()
        if len(dat)<100:continue
        tr,te=train_test_split(np.arange(len(dat)),test_size=.25,random_state=42)
        for name,model in models().items():
            model.fit(dat.iloc[tr][features],dat.iloc[tr][target]);pred=model.predict(dat.iloc[te][features]);actual=dat.iloc[te][target]
            results['outcomes'][target+'_'+name]={'n_train':len(tr),'n_test':len(te),
                'mae_g_cm2':float(mean_absolute_error(actual,pred)),
                'rmse_g_cm2':float(np.sqrt(mean_squared_error(actual,pred))),
                'r2':float(r2_score(actual,pred))}
            joblib.dump({'model':model,'features':features,'outcome':target,'data_type':'cross_sectional'},outdir/f'bone_{target}_{name}.joblib')
    (outdir/'bone_results.json').write_text(json.dumps(results,indent=2));return results
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/bone.yaml');ap.parse_args()
    r=project_root();src=r/'data'/'processed'/'bone_density'/'nhanes_bone.csv'
    if not src.exists():raise SystemExit('REAL NHANES DXA FILES NOT AVAILABLE. Run data.download and data.prepare; no synthetic BMD training.')
    print(json.dumps(train(src,r/'outputs'/'checkpoints'),indent=2))
