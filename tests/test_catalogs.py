import csv
from pathlib import Path
from gravity_ai.data.catalog_loader import validate,CATALOG_NAMES,project_root
from gravity_ai.data.prior_research import coverage_summary
from gravity_ai.data.validation import group_split

def test_all_five_inputs_present():
    root=project_root()/'data'/'catalogs'
    assert all((root/f).exists() for f in CATALOG_NAMES)
    assert (root/'embraceplus_earth_human_data_catalog.xlsx').exists()

def test_catalog_expected_counts():
    counts=validate()['counts']
    assert counts=={'signals':9,'biomarkers':26,'dataset_sources':11,'metric_mappings':16}

def test_previous_study_manifest_integrated():
    x=coverage_summary();assert x['source_records']==40 and x['variable_mappings']==102

def test_group_split_no_leakage():
    groups=['a']*3+['b']*3+['c']*3+['d']*3+['e']*3
    tr,te=group_split(groups)
    assert set(groups[i] for i in tr).isdisjoint(set(groups[i] for i in te))
