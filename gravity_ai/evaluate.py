"""Summarize actual completed training, without inventing missing health-model metrics.

For the spacecraft dynamics predictor this collects the result of the most recent fresh,
checkpoint-based evaluation (python -m gravity_ai.evaluate_dynamics). It does not retrain or
re-run anything itself. The archived 70-episode model's own training metrics are listed
separately; they were measured on a different benchmark and are not directly comparable.
"""
import argparse,json
from pathlib import Path
from gravity_ai.data.catalog_loader import project_root,validate

def _latest_dynamics(root):
    pointer=root/'outputs'/'runs'/'latest_dynamics.json'
    if not pointer.is_file():return None,'No completed dynamics-v2 run (python -m gravity_ai.train.dynamics)'
    run_dir=Path(json.loads(pointer.read_text(encoding='utf-8'))['run_dir'])
    ev=run_dir/'evaluation'/'evaluation.json'
    if not ev.is_file():return None,f'Run {run_dir.name} has no fresh evaluation yet (python -m gravity_ai.evaluate_dynamics --run {run_dir})'
    e=json.loads(ev.read_text(encoding='utf-8'))
    test=e['one_step']['test']
    return {'source':str(ev),'how':'checkpoint reloaded from disk and re-evaluated on frozen held-out test episodes',
            'run_id':e['run_id'],'checkpoint':e['checkpoint'],'data_type':e['data_type'],
            'test_rows':test['rows'],'test_episodes':test['episodes'],
            'one_step_test_mae':{m:{k:v['mae'] for k,v in test['models'][m].items() if 'mae' in v} for m in test['models']},
            'rollouts_test':{k:{m:{'ring_rpm_mae':v['models'][m]['ring_omega_rpm']['mae'],
                                   'gravity_g_mae':v['models'][m]['crew_gravity_magnitude_g']['mae']} for m in v['models']}
                             for k,v in e['rollouts'].items()}},None

def evaluate():
    root=project_root();cat=validate()
    out={'catalogs':cat['counts'],'trained_models':{},'missing':{}}
    archived=root/'outputs'/'metrics'/'dynamics_training.json'
    if archived.exists():
        out['trained_models']['dynamics_archived_v1']={**json.loads(archived.read_text()),
            'benchmark_note':'archived 70-episode, dt=1 s benchmark; different test set from dynamics-v2'}
    latest,why=_latest_dynamics(root)
    if latest:out['trained_models']['dynamics_v2']=latest
    else:out['missing']['dynamics_v2']=why
    for k,filename in {'physiology':'physiology_results.json','bone':'bone_results.json'}.items():
        p=root/'outputs'/'checkpoints'/filename
        if p.exists():out['trained_models'][k]=json.loads(p.read_text())
        else:out['missing'][k]='No completed training run or verified participant data available'
    target=root/'outputs'/'metrics'/'all_evaluation.json';target.parent.mkdir(exist_ok=True,parents=True);target.write_text(json.dumps(out,indent=2))
    return out
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--all',action='store_true');ap.parse_args()
    info=evaluate();print(json.dumps({'trained':list(info['trained_models']),'missing':info['missing']},indent=2))
