"""Manifest-driven, opt-in public data downloader with size protection.

Some catalog entries are landing pages, not direct data. No protected data are fetched.
Run in an internet-connected shell. Current build environment has no outbound DNS.
"""
from __future__ import annotations
import argparse,hashlib,json,re,sys
from pathlib import Path
from urllib.parse import urljoin,urlparse
import requests
from gravity_ai.data.catalog_loader import project_root
from gravity_ai.data.source_registry import sources

USER_AGENT='PersonalizedGravityResearch/0.1 (data audit; contact source administrator for bulk use)'


def fetch(url,dest,max_bytes=100_000_000,timeout=25):
    if not url.startswith('https://'):raise ValueError('HTTPS source required')
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256();size=0
    with requests.get(url,stream=True,timeout=timeout,headers={'User-Agent':USER_AGENT},allow_redirects=True) as r:
        r.raise_for_status()
        length=r.headers.get('Content-Length')
        if length and int(length)>max_bytes:raise ValueError(f'Refusing {length} bytes; max_bytes={max_bytes}')
        ct=r.headers.get('Content-Type','')
        if 'text/html' in ct and dest.suffix.lower() not in ('.htm','.html'):
            raise ValueError(f'HTML landing page instead of data file: {r.url}')
        with dest.open('wb') as f:
            for chunk in r.iter_content(65536):
                if not chunk:continue
                size+=len(chunk)
                if size>max_bytes:raise ValueError('Download exceeds configured size cap')
                h.update(chunk);f.write(chunk)
    return {'url':url,'destination':str(dest),'bytes':size,'sha256':h.hexdigest()}


def download_dataset(source_id,root=None,include_large=False,limit_subjects=9):
    records=sources()
    if source_id not in records:raise KeyError('Source not in user catalog: '+source_id)
    info=records[source_id]
    root=Path(root or project_root()/'data'/'raw')
    if source_id=='NHANES2013':
        result=[]
        for name in ['DEMO_H','BPX_H','DXXFEM_H','DXXSPN_H','PAXDAY_H','PAXHD_H']:
            url=f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/{name}.XPT'
            result.append(fetch(url,root/'nhanes'/f'{name}.XPT',max_bytes=25_000_000))
        return result
    if source_id=='MMASH':
        url='https://physionet.org/files/mmash/1.0.0/MMASH.zip'
        return [fetch(url,root/'mmash'/'MMASH.zip',max_bytes=50_000_000)]
    if source_id=='E4-STRESS-2025':
        # Purposefully download a documented small number of sessions, NOT 70MB by default.
        # PhysioNet redirects public files; preserve original URLs for provenance.
        # Directory contents as documented in the original dataset publication.
        dataset='https://physionet.org/files/wearable-device-dataset/1.0.1/Wearable_Dataset'
        cohort=['S01','S03','S04','S05','S06','S08','S09','S10','S11','S13','S14','S15','S16','S17','S18']
        if not include_large:cohort=cohort[:limit_subjects]
        result=[]
        for subject in cohort:
            for name in ['HR.csv','EDA.csv','TEMP.csv','ACC.csv','IBI.csv','tags.csv']:
                url=f'{dataset}/STRESS/{subject}/{name}'
                try:result.append(fetch(url,root/'empatica_e4'/'STRESS'/subject/name,max_bytes=5_000_000))
                except Exception as e:result.append({'url':url,'status':'failed','reason':str(e)[:200]})
        return result
    if source_id=='BIDMC':
        # PhysioNet contains waveform CSVs; individual sample downloading is opt-in.
        if not include_large:return [{'id':source_id,'status':'metadata_only','reason':'Use --include-large for individual waveforms; no blanket clinical-data training.'}]
        base='https://physionet.org/files/bidmc/1.0.0/bidmc_csv'
        return [fetch(f'{base}/bidmc_{i:02d}_Numerics.csv',root/'bidmc'/f'bidmc_{i:02d}_Numerics.csv',max_bytes=1_000_000) for i in range(1,6)]
    # Landing pages are not direct public downloads; no scraping bypass or blind bulk pulls.
    return [{'id':source_id,'status':'metadata_only_or_manual','reason':info['Access'], 'source_url':info['Primary URL']}]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--from-catalog',action='store_true')
    ap.add_argument('--public-only',action='store_true')
    ap.add_argument('--dataset',default='pilot',help='pilot, all, or a catalog source ID')
    ap.add_argument('--include-large',action='store_true')
    ap.add_argument('--limit-subjects',type=int,default=9)
    args=ap.parse_args()
    ids=['E4-STRESS-2025','NHANES2013'] if args.dataset=='pilot' else (list(sources()) if args.dataset=='all' else [args.dataset])
    records=[]
    for dataset in ids:
        try:records.extend(download_dataset(dataset,include_large=args.include_large,limit_subjects=args.limit_subjects))
        except Exception as e: records.append({'id':dataset,'status':'failed','reason':str(e)[:350], 'source_url':sources()[dataset]['Primary URL']})
    out=project_root()/'data'/'manifests'/'download_report.json'
    out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(records,indent=2))
    done=sum(x.get('bytes',0)>0 for x in records)
    print(json.dumps({'downloaded_files':done,'total_records':len(records),'report':str(out)},indent=2))
    if not done:print('No data downloaded. Connect to the Internet or place authorized raw files under data/raw; NO human training is performed using catalog rows.',file=sys.stderr)
if __name__=='__main__':main()
