from gravity_ai.data.catalog_loader import load_catalogs,CATALOG_NAMES

def coverage():
    return {r['Unified key']:r for r in load_catalogs()[CATALOG_NAMES[3]]}
