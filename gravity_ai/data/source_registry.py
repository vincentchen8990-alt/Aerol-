"""Dataset discovery is driven exclusively by the user's earth_human_dataset_sources.csv."""
from gravity_ai.data.catalog_loader import load_catalogs,CATALOG_NAMES

def sources():
    return {r['ID']:r for r in load_catalogs()[CATALOG_NAMES[2]]}
