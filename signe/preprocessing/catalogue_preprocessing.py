#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from signe.preprocessing.signalec.rpa_to_curb import (
    string_to_regulations,
    parking_regex,
    no_parking_regex,
    # s3r_regex,
    standing_regex,
    no_standing_regex
)


def main(catalogue):

    catalogue_copy = catalogue.copy()

    # 1 - Panonceaux
    # panonceaux = catalogue_copy[
    #     catalogue_copy.CODE_RPA.str.startswith('PX') |
    #     catalogue_copy.DESCRIPTION_RPA.str.startswith('PANONCEAU')
    # ].copy()
    catalogue_copy = catalogue_copy[
        ~(catalogue_copy.CODE_RPA.str.startswith('PX') |
          catalogue_copy.DESCRIPTION_RPA.str.startswith('PANONCEAU'))
    ]

    # 2 - Exclusive parking sign
    stationnement = pd.concat([
        catalogue_copy[
            catalogue_copy.DESCRIPTION_RPA.str.match(no_parking_regex)
        ].copy(),
        catalogue_copy[
            catalogue_copy.DESCRIPTION_RPA.str.match(parking_regex)
        ].copy(),
        catalogue_copy[
            catalogue_copy.DESCRIPTION_RPA.str.match(standing_regex)
        ].copy(),
        catalogue_copy[
            catalogue_copy.DESCRIPTION_RPA.str.match(no_standing_regex)
        ].copy()
    ])

    # 3 - Convert parking
    stationnement['CurbLR'] = stationnement.DESCRIPTION_RPA.apply(
        string_to_regulations
    )

    return stationnement
