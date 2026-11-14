"""Primary four user CSV catalogs; these contain metadata, NOT patient measurements."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path

CATALOG_NAMES = (
 'embraceplus_watch_signals.csv','embraceplus_watch_biomarkers.csv',
 'earth_human_dataset_sources.csv','earth_human_metric_mapping.csv')
EXPECTED_HEADERS={
 CATALOG_NAMES[0]:['Signal / feature','Layer','Raw field / units','Sample timing','Availability','Use in gravity-health research','Earth analog','Official source'],
 CATALOG_NAMES[1]:['Signal / report','Area','Vendor output family','Units','Availability','Important interpretation','Earth counterpart','Official source'],
 CATALOG_NAMES[2]:['ID','Dataset','Watch/device match','Cohort','Available metrics','Access','Sampling / details','Important limits','Primary URL'],
 CATALOG_NAMES[3]:['Unified key','Target feature','Layer','Earth reference IDs','Coverage status','Caveat'],
}

def project_root():return Path(__file__).resolve().parents[2]

def load_catalogs(directory=None):
    directory=Path(directory or project_root()/'data'/'catalogs')
    result={}
    for filename in CATALOG_NAMES:
        path=directory/filename
        if not path.is_file():raise FileNotFoundError(f'Catalog missing: {path}')
        with path.open('r',encoding='utf-8-sig',newline='') as fp:
            reader=csv.DictReader(fp)
            headers=reader.fieldnames
            if headers!=EXPECTED_HEADERS[filename]:raise ValueError(f'{filename}: unexpected headers {headers}; expected {EXPECTED_HEADERS[filename]}')
            rows=list(reader)
        if not rows: raise ValueError(f'{filename}: no data rows')
        result[filename]=rows
    return result

def validate(directory=None,output=None):
    data=load_catalogs(directory)
    directory=Path(directory or project_root()/'data'/'catalogs')
    report={'catalog_type':'metadata_only_not_training_measurements','files':{},'sources':[], 'features':[], 'mapping_coverage':[], 'warnings':[]}
    for filename,rows in data.items():
        path=directory/filename
        report['files'][filename]={'rows':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'missing_cells':sum(not str(v or '').strip() for row in rows for v in row.values())}
    ids=set()
    for row in data[CATALOG_NAMES[2]]:
        if row['ID'] in ids:raise ValueError('Duplicate dataset source ID: '+row['ID'])
        ids.add(row['ID'])
        report['sources'].append({'id':row['ID'],'name':row['Dataset'],'url':row['Primary URL'],
                                  'access':row['Access'],'category':'source_record'})
    for row in data[CATALOG_NAMES[0]]:
        report['features'].append({'name':row['Signal / feature'],'layer':'raw_watch','availability':row['Availability'],
                                  'description':row['Raw field / units'],'source_url':row['Official source']})
    for row in data[CATALOG_NAMES[1]]:
        report['features'].append({'name':row['Signal / report'],'layer':'biomarker','availability':row['Availability'],
                                  'description':row['Important interpretation'],'source_url':row['Official source']})
    for row in data[CATALOG_NAMES[3]]:
        report['mapping_coverage'].append({'key':row['Unified key'],'sources':row['Earth reference IDs'],'coverage':row['Coverage status']})
    report['counts']={'signals':len(data[CATALOG_NAMES[0]]),'biomarkers':len(data[CATALOG_NAMES[1]]),
                      'dataset_sources':len(data[CATALOG_NAMES[2]]),'metric_mappings':len(data[CATALOG_NAMES[3]])}
    report['warnings'].append('Catalogs contain metadata and source URLs. Do not use catalog row counts as human sample counts.')
    report['warnings'].append('Restricted and public data are separate; preserve participant and session identity.')
    if output:
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--directory')
    args=parser.parse_args()
    output=project_root()/'data'/'manifests'/'catalog_validation.json'
    report=validate(args.directory,output)
    print(json.dumps({'counts':report['counts'],'output':str(output),'status':'validated; metadata only'},indent=2))
