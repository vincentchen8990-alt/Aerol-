"""Train on actual E4 sessions only. Never train on the source catalog rows."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from gravity_ai.data.catalog_loader import project_root
from gravity_ai.data.validation import group_split
from gravity_ai.models.physiology import ridge_model,boosting_model
import joblib

def train(data_path,output_dir):
    df=pd.read_csv(data_path)
    for field in ('session_id','subject_id','hr_mean'):
        if field not in df:raise ValueError('Missing real E4 window field '+field)
    if df.subject_id.nunique()<5:raise ValueError('At least five independent real subjects required for honest holdout')
    df=df.sort_values(['session_id','window_start_utc_s']).copy()
    df['next_hr_mean']=df.groupby('session_id')['hr_mean'].shift(-1)
    df=df.dropna(subset=['next_hr_mean'])
    fields=[s for s in ('hr_mean','hr_std','eda_mean','eda_std','temp_mean','temp_std','acc_norm_mean_g','acc_norm_std_g') if s in df]
    X=df[fields];y=df['next_hr_mean'].values;groups=df.subject_id.values
    tr,te=group_split(groups,test_size=.25,seed=42)
    if len(set(groups[tr]))<2:raise ValueError('Too few train subjects')
    result={'data_type':'measured_E4_not_gravity_response','n_samples':len(df),'participants_total':int(df.subject_id.nunique()),
            'train_subjects':len(set(groups[tr])),'test_subjects':len(set(groups[te])),'features':fields,'models':{}}
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    for name,model in [('ridge',ridge_model()),('boosting',boosting_model())]:
        model.fit(X.iloc[tr],y[tr]);pred=model.predict(X.iloc[te]);
        result['models'][name]={'test_mae_bpm':float(mean_absolute_error(y[te],pred)),
                                'test_rmse_bpm':float(np.sqrt(mean_squared_error(y[te],pred))),
                                'test_r2':float(r2_score(y[te],pred))}
        joblib.dump({'model':model,'features':fields,'data_source':'real_E4'},output_dir/f'physiology_{name}.joblib')
    (output_dir/'physiology_results.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/physiology.yaml');args=ap.parse_args()
    root=project_root();data=root/'data'/'processed'/'wearable'/'e4_windows.csv'
    if not data.exists():raise SystemExit('REAL E4 MEASUREMENTS NOT AVAILABLE. Run data.download and data.prepare with network access; no fake training.')
    print(json.dumps(train(data,root/'outputs'/'checkpoints'),indent=2))
