#!/usr/bin/env python
# coding:  utf-8

import re

# replace with regex
periods_value = {
    "1 MARS AU 1 DEC": [{"from": "03-01", "to": "12-01"}],
    "1ER MARS 1ER DEC": [{"from": "03-01", "to": "12-01"}],
    "1ER MARS - 1ER DÉC": [{"from": "03-01", "to": "12-01"}],
    "1ER MARS - 1ER  DÉC": [{"from": "03-01", "to": "12-01"}],
    "1ER MARS AU 1ER DECEMBRE": [{"from": "03-01", "to": "12-01"}],
    "1ER MARS AU 1ER DEC": [{"from": "03-01", "to": "12-01"}],
    "MARS 01 A DEC. 01": [{"from": "03-01", "to": "12-01"}],
    "1 MARSL AU 1 DEC": [{"from": "03-01", "to": "12-01"}],
    "1MARS AU 1 DEC.": [{"from": "03-01", "to": "12-01"}],
    "15 MARS AU 15 NOV": [{"from": "03-15", "to": "11-15"}],
    "15 MARS AU 15 NOVEMBRE": [{"from": "03-15", "to": "11-15"}],
    "1 AVRIL AU 30 SEPT": [{"from": "04-01", "to": "09-30"}],
    "1 AVRIL AU 15 OCT": [{"from": "05-01", "to": "10-15"}],
    "1 AVRIL AU 31 OCT": [{"from": "05-01", "to": "10-31"}],
    "1 AVRIL AU 1 NOVEMBRE": [{"from": "05-01", "to": "11-01"}],
    "1 AVRIL AU 15 NOV": [{"from": "05-01", "to": "11-15"}],
    "1 AVRIL AU 15 NOVEMBRE": [{"from": "05-01", "to": "11-15"}],
    "1 AVRIL AU 30 NOV": [{"from": "04-01", "to": "11-30"}],
    "1ER AVRIL - 30 NOV": [{"from": "04-01", "to": "11-30"}],
    "1 AVRIL AU 1 DEC": [{"from": "04-01", "to": "12-01"}],
    "1 AVIL AU 1 DEC": [{"from": "04-01", "to": "12-01"}],
    "1 AVRIL ET 1 DEC": [{"from": "04-01", "to": "12-01"}],
    "1AVRIL AU 1 DEC": [{"from": "04-01", "to": "12-01"}],
    "1AVRIL AU 1DEC": [{"from": "04-01", "to": "12-01"}],
    "1ER AVRIL AU 1ER DEC": [{"from": "04-01", "to": "12-01"}],
    "AVRIL 01 A DEC. 01": [{"from": "04-01", "to": "12-01"}],
    "1 AVRILS AU 1 DEC": [{"from": "04-01", "to": "12-01"}],
    "1 AVRIL  AU 1 DEC": [{"from": "04-01", "to": "12-01"}],
    "15 AVRIL AU 15 OCTOBRE": [{"from": "04-15", "to": "10-15"}],
    "15 AVRIL AU 1 NOV": [{"from": "05-15", "to": "11-01"}],
    "15 AVRIL AU 1ER NOV.": [{"from": "04-15", "to": "11-01"}],
    "15 AVRIL AU 1 NOVEMBRE": [{"from": "04-15", "to": "11-01"}],
    "15 AVRIL AU 15 NOVEMBRE": [{"from": "04-15", "to": "11-15"}],
    "15 AVRIL AU 1ER DEC": [{"from": "05-15", "to": "12-01"}],
    "1MAI AU 1 SEPT": [{"from": "05-01", "to": "09-01"}],
    "1MAI AU 1OCT": [{"from": "05-01", "to": "10-01"}],
    "1 MAI AU 1 NOV": [{"from": "06-01", "to": "11-01"}],
    "15 MAI AU 15 OCT": [{"from": "05-15", "to": "10-15"}],
    "15 MAI AU 15 SEPT": [{"from": "05-15", "to": "09-15"}],
    "1 JUIN AU 1 OCT": [{"from": "06-01", "to": "10-01"}],
    "21 JUIN AU 1 SEPT": [{"from": "06-21", "to": "09-01"}],
    "30 JUIN AU 30 AOUT": [{"from": "06-30", "to": "08-30"}],
    "15 AOUT - 28 JUIN": [{"from": "08-15", "to": "06-28"}],
    "20 AOÛT AU 30 JUIN": [{"from": "08-20", "to": "06-30"}],
    "1 SEPT. AU 23 JUIN": [{"from": "09-01", "to": "06-23"}],
    "SEPT A JUIN": [{"from": "09-01", "to": "06-30"}],
    "SEPT À JUIN": [{"from": "09-01", "to": "06-30"}],
    "SEPT. A JUIN": [{"from": "09-01", "to": "06-30"}],
    "1 SEPT. AU 30 JUIN": [{"from": "09-01", "to": "06-30"}],
    "1 SEPT. AU 31 MAI": [{"from": "09-01", "to": "05-31"}],
    "1 NOV. AU 31 MARS": [{"from": "11-01", "to": "03-31"}],
    "1 NOV. AU 1 AVRIL": [{"from": "11-01", "to": "04-01"}],
    "1 NOVEMBRE AU 15 AVRIL": [{"from": "11-01", "to": "04-15"}],
    "1 NOV. AU 1 MAI": [{"from": "11-01", "to": "05-01"}],
    "15 NOV. AU 15 MARS": [{"from": "11-15", "to": "03-15"}],
    "15 NOV. AU 1 AVRIL": [{"from": "11-15", "to": "04-01"}],
    "16 NOV. AU 14 MARS": [{"from": "11-16", "to": "03-14"}],
    "30 NOV - 1ER AVRIL": [{"from": "11-30", "to": "04-01"}],
    "1 DEC. AU 1 MARS": [{"from": "12-01", "to": "03-01"}],
    "1ER DECEMBRE AU 1ER MARS": [{"from": "12-01", "to": "03-01"}],
    "1 DEC. AU 1 AVRIL": [{"from": "12-01", "to": "04-01"}]
}
period_text = '|'.join(periods_value.keys())

DAYS = ['mo', 'tu', 'we', 'th', 'fr', 'sa', 'su']
day_values = {
    "mo": ["LUNDI", "LUN\\.", "LUN"],
    "tu": ["MARDI", "MAR\\.", "MAR"],
    "we": ["MERCREDI", "MER\\.", "MER"],
    "th": ["JEUDI", "JEU\\.", "JEU"],
    "fr": ["VENDREDI", "VEN\\.", "VEN", "VEMDREDI"],
    "sa": ["SAMEDI", "SAM\\.", "SAM"],
    'su': ["DIMANCHE", "DIM\\.", "DIM"]
}
day_text = '|'.join([item for sublist in day_values.values() for item in sublist])

## REGEX
standing_regex = r'^A\s.*'
no_standing_regex = r'^\\A\s.*'
no_parking_regex = r'(?: /[P,p] )|(?: ^\\P\b)|(?: (?: INT(?: \.|ERDICTION){0,1}){1}.*\bSTAT(?: \.|IONNEMENT){0,1}\b.*)|(?: .*\bSTAT(?: \.|IONNEMENT){0,1}\b.*(?: INT(\.|ERDI){0,1}|R[E,É]SERV[E,É]){1})|.*VOIE R[E,É]SERV[E,É]E.*'
parking_regex = r'(?=^P |STATIONNEMENT|STAT|STAT\.)(?=(?!{}))'.format(no_parking_regex)
livraison_regex = r'(?=.*{})(?=.*LIVRAISON.*)'.format(no_parking_regex)
s3r_regex =  r'(?: (?=.*{})(?=.*(?: \sSRRR\s*|\sS3R\s*|\sPERMIS\s*).*))|(?: (?=.*{})(?=.*(?: \sSRRR\s*|\sS3R\s*|\sPERMIS\s*).*))'.format(no_parking_regex, parking_regex)
reserve_regex = r'(?=(?: '+no_parking_regex+r')|(?: '+parking_regex+r'))(?=(?: .*(?: (?: R[E,É]SERV[É,E][E]{0,1})|(?: EXCEPT[É,E][E]{0,1})) (.*?)(\s*(?: $|\d+(?: min|h).*)|(?: '+day_text+r').*))|(?: (?: ^[\\P, P]|.*\d{1,2}H|.*[LUN,MAR,MER,JEU,VEN,SAM,DIM]) (\w+?) SEULEMENT.*))'

classes = {
    'autobus':  r'.*(?: AUTOBUS|BUS|D\'AUTOBUS|BIBLIOBUS).*',
    'electric':  r'.*(?: ELECTRIQUE|CHARGE).*',
    'handicap':  r'.*HANDI.*',
    'visitor':  r'.*VISITEUR.*',
    'permit':  s3r_regex,
    'police':  r'^(?!.*VISITEUR).* POLICE.*$',
    'city':  r'^(?!.*VISITEUR).* DE LA VILLE.*$',
    'taxi':  r'.*TAXI.*',
}
activities = {
    'loading':  livraison_regex,
    'parking':  parking_regex,
    'no parking':  no_parking_regex,
    'standing':  standing_regex,
    'no_standing':  no_standing_regex,
}
categories = {
    'permit parking':  s3r_regex,
    'street cleaning':  r'^.*(?: '+period_text+r').*$'
}

### TRANSFORM

def parse_days(day):
    """ Parse a days interval express by string to list of
   datetime.dayofweek values.

    Parameters
    ----------
    day :  string
        a string representing a day interval. Takes values from DAYS.

    Return
    ------
    list_of_days :  list of int
        int representation of the day interval.

    Example
    -------
    1. Joint interval of several days :
        >>> parse_days('lun-mer') ---> return [0, 1, 2]
    2. Interval of disjoint days :
        >>> parse_days('lun+mer') ---> return [0, 3]
    3. Interval of only one day :
        >>> parse_days('lun') ----> return [0]
    """
    if day == 'dim-sam':
        return list(range(0, 7))
    day = day.strip()
    if day in DAYS:
        return [DAYS.index(day)]

    if '-' in day:
        first = DAYS.index(day.split('-')[0].strip())
        last = DAYS.index(day.split('-')[-1].strip())

        return list(range(first, last+1))

    if '+' in day:
        return [DAYS.index(d.strip()) for d in day.split('+')]

def transform_day(description):
    """TODO
    """

    days = []

    day_text = '|'.join([item for sublist in day_values.values() for item in sublist])
    day_map = {}
    for k, v in day_values.items():
        for vv in v:
            day_map[vv.replace('\\', '')] = k

    days_rule = re.findall(f'({day_text})(.*?)({day_text})', description)
    for day_rule in days_rule:
        if day_rule :
            day1 = day_map[day_rule[0]]
            day2 = day_map[day_rule[2]]

            if re.findall('.*(A|À|AU|a|à|au|-).*', day_rule[1]):
                days.extend([DAYS[i] for i in parse_days(f'{day1}-{day2}')])
            elif re.findall('(ET|,{0,1}\s|et)', day_rule[1]) or day_rule[1] == '':
                days.append(day1)
                days.append(day2)
            else :
                print("day rules verbe unknown", day_rule[2], description)
    else :
        day_rules = re.findall(f'({day_text})', description)[len(days): ]
        while day_rules:
            days.append(day_map[day_rules.pop()])

    return days

def transform_hour(description):
    time_regexp='(?: (\d+)[H,h,: ](\d*))\s*?[Aaà@-]\s*?(?: (\d+)[H,h,: ](\d*))'

    hours = []

    hour_intervals = re.findall(time_regexp, description)
    for hour_interval in hour_intervals:
        interval_dict = {'from': '', 'to': ''}
        hour_first, min_first, hour_last, min_last = hour_interval
        interval_dict['from'] = f"{int(hour_first): 02}: {int(min_first): 02}" if min_first else f"{int(hour_first): 02}: 00"
        interval_dict['to'] = f"{int(hour_last): 02}: {int(min_last): 02}" if min_last else f"{int(hour_last): 02}: 00"
        if int(interval_dict['from'][: 2]) > int(interval_dict['to'][: 2]):
            hours.append({'from': interval_dict['from'], 'to': '23: 59'})
            hours.append({'from': '00: 00', 'to': interval_dict['to']})
        else:
            hours.append(interval_dict)

    return hours

def transform_effective_dates(description):
    effectives_dates = []

    periods = re.findall(f'.*({period_text}).*', description)
    for period in periods:
        effectives_dates.extend(periods_value[period])

    return effectives_dates


def transform_time_span(description, catalogue_prking=False):
    """TODO
    """
    time_spans = []

    if catalogue_prking:
        #catalogue = pd.read_csv('../data/catalogue_RPA_pkrng.csv')
        pass
    else:
        # TODO :  Regex to find several time_span description
        detect_timespans = re.compile(r"(\d+[h,: ]\d*[\s,à,@,-]\d+[h,: ]\d*.*?(?: "+day_text+r").*?(?: "+day_text+r"))", flags=re.IGNORECASE)
        start_timespans = [x.start() for x in detect_timespans.finditer(description)]
        for i in range(len(start_timespans)):
            if i == len(start_timespans) - 1:
                description_iter = description[start_timespans[i]: ]
            else:
                description_iter = description[start_timespans[i]: start_timespans[i+1]]
            time_span = {}
            # days
            days = transform_day(description_iter)
            if days:
                time_span["daysOfWeek"] = {'days': days}

            # hours
            time_span["timesOfDay"] = transform_hour(description_iter)

            # effective dates
            if transform_effective_dates(description):
                time_span["effectiveDates"] = transform_effective_dates(description)

            time_spans.append(time_span)

        # not following the multiregex
        if not time_spans:
            time_span = {}
            # days
            days = transform_day(description)
            if days:
                time_span["daysOfWeek"] = {'days': days}

            # hours
            time_span["timesOfDay"] = transform_hour(description)

            # effective dates
            if transform_effective_dates(description):
                time_span["effectiveDates"] = transform_effective_dates(description)

            time_spans.append(time_span)

        #check if timespans empty
        empty=True
        for timespan in time_spans:
            for time_rule, time_values in timespan.items():
                if time_values:
                    empty = False
        if empty:
            time_spans=[]

    return time_spans

def tranform_rules(description, default_max_stay=15):

    rule = {'activity': '', 'priorityCategory': ''}
    for activity, finder in activities.items():
        regex = re.compile(finder, re.IGNORECASE)
        if regex.match(description):
            rule['activity'] = activity
            rule['priorityCategory'] = activity
            break

    for category, finder in categories.items():
        regex = re.compile(finder, flags=re.IGNORECASE)
        if regex.match(description):
            rule['priorityCategory'] = category
            break

    #si no parking mais reserve alors parking pour reservé
    regex = re.compile(reserve_regex, flags=re.IGNORECASE)
    if rule['activity'] == 'no parking' and regex.match(description):
        rule['activity'] = 'parking'

    if rule['activity'] == 'no standing' and regex.match(description):
        rule['activity'] = 'standing'

    # check for max_stays
    regex = re.compile(r'(?: .*?(\d+)\s*MIN(?: UTES){0,1}.*)|.*DEBARCADERE.*', flags=re.IGNORECASE)
    if regex.match(description):
        rule['maxStay'] = regex.findall(description)[0] if regex.findall(description) else default_max_stay

    return rule

def transform_user_classes(description):

    user_classes = []

    # S3R
    if re.compile(s3r_regex, flags=re.IGNORECASE).match(description):
        user_classes.append({'classes': ['permit'], 'subclasses': ['to_determine']})

    # Other (automatic detection)
    regex = re.compile(reserve_regex, flags=re.IGNORECASE)
    for detected_user_class in regex.findall(description):
        for user_class, finder in classes.items():
            user = {'classes': []}
            regex = re.compile(finder, flags=re.IGNORECASE)
            if regex.match(''.join(detected_user_class)):
                user['classes'] = [user_class]
                user_classes.append(user)

    return user_classes

def string_to_regulations(description):
    regulations = []

    # We make the hypothesis that there is only one regulation by pan
    regulations.append({
        'rule': tranform_rules(description),
    })
    if transform_user_classes(description):
        regulations.append({
            'userClasses': transform_user_classes(description),
        })
    if transform_time_span(description):
        regulations.append({
            'timeSpans': transform_time_span(description),
        })

    return regulations