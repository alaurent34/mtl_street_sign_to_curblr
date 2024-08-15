#!/usr/bin/env python
# coding: utf-8

# TODO: Automatise pannonceaux regulation detection
PANONCEAUX = {
    'PX-PD': {
        'rule': {'maxStay': '15'}
    },
    'PX-EU': {
        "userClasses": [{
            "classes": ["garderie"]
        }]
    },
    'PX-HC': {
        "userClasses": [{
            "classes": ["handicap"]
        }]
    },
    'PX-PH': {},
    'PX-AH': {
        "timeSpans": [{
            "effectiveDates": [
                  {"from": "11-01", "to": "04-15"}
                ],
        }],
    },
    'PX-PR': {
        "rule": {
                "activity": "parking",
                "priorityCategory": "permit parking",
            },
        "userClasses": [{
            "classes": ["permit"]
        }]
    },
    'PX-PL': {
        "rule": {
                "activity": "loading",
                "priorityCategory": "loading",
            }
    },
    'PX-AY': {
        "timeSpans": [{
            "effectiveDates": [
                  {"from": "11-01", "to": "04-15"}
                ],
        }],
    }
}
