from gravity_ai.data.catalog_loader import load_catalogs,CATALOG_NAMES

def feature_registry():
    tables=load_catalogs()
    return {'raw_signals':tables[CATALOG_NAMES[0]],'biomarkers':tables[CATALOG_NAMES[1]]}
