"""Previously researched verified dataset manifest (40 records) and dictionary (102).

The two CSVs are reference metadata; they do not grant access to raw private data.
"""
from pathlib import Path
import csv
from gravity_ai.data.catalog_loader import project_root

def load_reference(name):
    if name not in ('dataset_manifest.csv','data_dictionary.csv'):raise ValueError('Unexpected reference file')
    with (project_root()/'reference'/name).open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def coverage_summary():
    d=load_reference('dataset_manifest.csv');m=load_reference('data_dictionary.csv')
    return {'source_records':len(d),'variable_mappings':len(m),'measured_control_sources':[
        {'id':r['manifest_id'],'title':r['dataset_title'],'verified':r['verification_status']}
        for r in d if r['data_origin'].lower().startswith('measured')][:6],
        'note':'Manifest coverage is not participant data; original sample archive is not embedded.'}
