"""Prepare only participant-level measurements actually found on disk."""
import argparse,json
from pathlib import Path
from gravity_ai.data.catalog_loader import project_root,validate
from gravity_ai.data.empatica import load_sessions
from gravity_ai.data.nhanes import load_nhanes_bone

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',default='all');args=ap.parse_args()
    root=project_root();validate(output=root/'data'/'manifests'/'catalog_validation.json')
    report={'data_type':'real_measurements_only','prepared':{},'unavailable':{}}
    if args.dataset in ('all','empatica_e4'):
        try:
            df=load_sessions(root/'data'/'raw'/'empatica_e4')
            out=root/'data'/'processed'/'wearable'/'e4_windows.csv';out.parent.mkdir(parents=True,exist_ok=True);df.to_csv(out,index=False)
            report['prepared']['empatica_e4']={'rows':len(df),'participants':int(df.subject_id.nunique()),'path':str(out)}
        except (FileNotFoundError,ValueError) as e:report['unavailable']['empatica_e4']=str(e)
    if args.dataset in ('all','nhanes'):
        try:
            df,features,outcomes=load_nhanes_bone(root/'data'/'raw'/'nhanes')
            out=root/'data'/'processed'/'bone_density'/'nhanes_bone.csv';out.parent.mkdir(parents=True,exist_ok=True)
            df[['SEQN',*features,*outcomes]].to_csv(out,index=False)
            report['prepared']['nhanes']={'rows':len(df),'features':features,'outcomes':outcomes,'path':str(out)}
        except (FileNotFoundError,ValueError) as e:report['unavailable']['nhanes']=str(e)
    dest=root/'data'/'manifests'/'preparation_report.json';dest.parent.mkdir(exist_ok=True,parents=True);dest.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
