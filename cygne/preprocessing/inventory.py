""" Process inventory data
"""
import json
import logging
import argparse
import glob
import os

import geopandas as gpd
from cygne.core.inventory import Inventory
from cygne.io.mtl_opendata import read_mtl_open_data

logger = logging.getLogger(__name__)


def get_latest_file(directory: str, pattern: str) -> str:
    """Find the latest file matching a pattern in a directory."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        raise FileNotFoundError(
            f"No matching file found for pattern '{pattern}' in '{directory}'. "
            "Please download the data from Montreal Open Data Portal."
        )
    return max(files, key=os.path.getctime)


def main():
    """ Main
    """
    parser = argparse.ArgumentParser(description='Inventory processing')
    parser.add_argument('--inventaire', type=str, help='Path to inventaire file')
    parser.add_argument('--support', type=str, help='Path to support file')
    parser.add_argument('--panneau', type=str, help='Path to panneau file')
    parser.add_argument('--period', type=str, help='Path to period file')
    args, _ = parser.parse_known_args()

    data_dir = './data/inventaire'

    try:
        inv_path = args.inventaire or get_latest_file(data_dir, 'inventaire_lapi_*.geojson')
        sup_path = args.support or get_latest_file(data_dir, 'rp_support_*.geojson')
        pan_path = args.panneau or get_latest_file(data_dir, 'rp_panneau_*.geojson')
        per_path = args.period or get_latest_file(data_dir, 'rp_panneau_periode_*.geojson')
    except FileNotFoundError as e:
        logger.error(e)
        return

    logger.info('Query data')
    logger.info(f"Using inventaire: {inv_path}")
    inventaire = gpd.read_file(inv_path, encoding='utf-8')
    support = gpd.read_file(sup_path, encoding='utf-8')
    panneau = gpd.read_file(pan_path, encoding='utf-8')
    period = gpd.read_file(per_path, encoding='utf-8')

    geobase = read_mtl_open_data(
        'https://data.montreal.ca/dataset/' +
        '984f7a68-ab34-4092-9204-4bdfcca767c5/' +
        'resource/9d3d60d8-4e7f-493e-8d6a-dcd040319d8d/download/geobase.json'
    )
    geobase = geobase.to_crs('epsg:32188')
    support = support.to_crs('epsg:32188')

    df = inventaire.join(
        support.set_index('parentglobalid'),
        on='globalid',
        lsuffix='_inventaire',
        how='right'
    ).join(
        panneau.set_index('parentglobalid'),
        on='globalid',
        rsuffix='_panneau',
        how='right'
    ).join(
        period.set_index('parentglobalid'),
        on='globalid_panneau',
        rsuffix='_period',
        how='left'
    )

    # FIX SRRR student's incorrect coding.
    df.loc[~df.RegVehSRRR.isna() &
           (df.RegVehSRRR != ''), 'RegVehExcept'] = 'oui'

    logger.info("Create panels collection")
    panc = Inventory.from_inventory(df)
    logger.info("Enrich panels location with geobase info")
    panc.enrich_with_roadnetwork(geobase)
    logger.info("Group panels with street and side")

    logger.info("Testing chaining of signs")
    pb_pans = panc.test_chaining()
    logger.info("This signs were causing problems : %s", pb_pans)

    logger.info("Creating CurbLR")
    curlr = panc.to_curblr()
    with open('./test_inventaire.curblr.geojson', 'w', encoding='utf-8') as f:
        json.dump(curlr, f, indent=4, ensure_ascii=False)

    logger.info('Done')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
