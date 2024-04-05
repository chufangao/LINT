import pandas as pd
import numpy as np
import copy
import datetime
import xml.etree.ElementTree
from tqdm.auto import tqdm
import multiprocessing
import pickle
import torch
from sentence_transformers import SentenceTransformer


def dictify(r, root=True, str_to_replace=''):
    if root:
        return {r.tag : dictify(r, root=False, str_to_replace=str_to_replace)}
    if r.findall("./*") == []:
        return r.text

    d=copy.copy(r.attrib)
    # if r.text:
    #     d["text"]=r.text
    for x in r.findall("./*"):
        if x.tag.replace(str_to_replace, '') not in d:
            d[x.tag.replace(str_to_replace, '')]=[]
        d[x.tag.replace(str_to_replace, '')].append(dictify(x, root=False, str_to_replace=str_to_replace))
    return d

def parse_drug_dict(d):
    important_vars = ['type', 'created', 'name', 'description', 'state', 
                      'indication', 'pharmacodynamics', 'mechanism-of-action', 
                      'toxicity', 'metabolism', 'absorption', 'half-life', 
                      'synonyms', 'products','international-brands',
                      'dosages', 'atc-codes', 'targets']

    new_d = {}
    for var in important_vars:
        if var not in d.keys():
            new_d[var] = None
        elif type(d[var]) == list and len(d[var]) == 1:
            if d[var][0] is not None:
                new_d[var] = d[var][0]
            else: new_d[var] = []
        else:
            new_d[var] = d[var]
    
    # Process texts
    for text_key in ['description', 'indication', 'pharmacodynamics', 'mechanism-of-action', 'toxicity', 'metabolism', 'absorption', 'half-life', 'state']:
        if text_key not in d.keys():
            new_d[text_key] = 'None'
        elif type(d[text_key]) == list and d[text_key][0] is None:
            new_d[text_key] = 'None'
    # =================== Process synonyms ===================
    if len(new_d['synonyms']) != 0:
        new_d['synonyms'] = new_d['synonyms']['synonym']
    # =================== Process products ===================
    if len(new_d['products']) != 0:
        products_list = [_['name'][0] for _ in new_d['products']['product']]
        products_list = list(set(products_list))
        new_d['products'] = products_list
    # =================== Process products ===================
    if len(new_d['international-brands']) != 0:
        products_list = [_['name'][0] for _ in new_d['international-brands']['international-brand']]
        products_list = list(set(products_list))
        new_d['international-brands'] = products_list
    # =================== Process targets ===================
    if len(new_d['targets']) != 0:
        products_list = [_['name'][0] for _ in new_d['targets']['target']]
        products_list = list(set(products_list))
        new_d['targets'] = products_list
    # =================== Process created (date) ===================
    # let's just convert to years from 1970
    years = datetime.datetime.fromisoformat(new_d['created']).timestamp()/(60*60*24*365)
    new_d['created'] = years
    # # =================== Process atc code ===================
    # new_d['atc-codes'] = new_d['atc-codes']['atc-code'][0]['level']
    # =================== Process dosages ===================
    if len(new_d['dosages']) > 0:
        new_d['dosages'] = new_d['dosages']['dosage']

    return new_d

def nctid2label_dict2(outcome2label_file="IQVIA/outcome2label.txt", nctid2outcome_file="IQVIA/trial_outcomes_v1.csv"):
    outcome2label = pd.read_csv(outcome2label_file, sep='\t', engine='python', header=None)
    outcome2label.columns=['trialOutcome', 'label']
    nctid2outcome = pd.read_csv(nctid2outcome_file, sep=',', engine='python')
    nctid2label = nctid2outcome.merge(outcome2label, on='trialOutcome', how='left')
    return {nctid2label['studyid'].values[i] : nctid2label['label'].values[i] for i in range(len(nctid2label))}

def load_disease2icd2(disease_file='data/diseases.csv'):
    data = pd.read_csv(disease_file)
    data = data[data['icd']!='None']
    disease2icd = {data['disease'].values[i] : eval(data['icd'].values[i]) for i in range(len(data))}
    return disease2icd

def load_country2continent(countries_file='data/countries.csv'):
    d = pd.read_csv(countries_file)
    country2continent = {d['Country'].values[i] : d['Continent'].values[i] for i in range(len(d))}
    return country2continent

def trial_date2float(year_str):
    ## assumes year is of format:
    ## "Month day, year" e.g. "October 2, 1999" OR
    ## "Month year" e.g. "October 1999"
    if ',' in year_str:
        return datetime.datetime.strptime(year_str, '%B %d, %Y').timestamp()/(60*60*24*365)
    else:
        return datetime.datetime.strptime(year_str, '%B %Y').timestamp()/(60*60*24*365)

def process_age(age_str):
    # assumes age is of type "NUM UNIT" e.g. "89 years" or "N/A"
    if age_str == 'N/A': 
        return "None"
    
    num, unit = age_str.lower().split(' ')
    num = eval(num)
    # convert to year
    if unit in ['second', 'seconds']: factor = 365*24*60*60.
    elif unit in ['minute', 'minutes']: factor = 365*24*60.
    elif unit in ['hour', 'hours']: factor = 365*24.
    elif unit in ['day', 'days']: factor = 365.
    elif unit in ['year', 'years']: factor = 1.
    elif unit in ['month', 'months']: factor = 12.
    elif unit in ['week', 'weeks']: factor = 52.
    else:
        print(unit)
        return "None"

    years = num / factor
    
    # ages taken from https://nexus.od.nih.gov/all/2022/04/11/fy-2021-data-on-age-at-enrollment-in-clinical-research-now-available-by-rcdc-category/
    if 0 <= years < 6:
        return "Child"
    elif 6 <= years < 18:
        return "Adolescent"
    elif 18 <= years < 65:
        return "Adult"
    else: # 65+
        return "Older_Adult"

def clean_protocol(protocol):
    protocol_split = protocol.lower().split('\n')
    return [x.strip() for x in protocol_split if len(x.strip())>0]

def parse_trial_dict(d, nctid2label, disease2icd, country2continent):
    d = d['clinical_study']
    important_vars = ['id_info', 'brief_summary', 'overall_status', 'why_stopped',
                        'phase', 'study_type', 'location_countries']
    new_d = {}
    for var in important_vars:
        if var not in d.keys():
            new_d[var] = []
        elif type(d[var]) == list and len(d[var]) == 1:
            if d[var][0] is not None:
                new_d[var] = d[var][0]
            else: new_d[var] = []
        else:
            new_d[var] = d[var]

    # =================== check phase = ===================
    new_d['phase'] = new_d['phase'].lower() if 'phase' in d.keys() else 'None'
        
    # =================== check study type ===================
    if new_d['study_type'] != 'Interventional': 
        return None, 'non-Interventional'

    # =================== Process id info ===================
    new_d['id_info'] = new_d['id_info']['nct_id'][0]

    # =================== Process text ===================
    new_d['brief_summary'] = new_d['brief_summary']['textblock'][0]

    # # =================== Process locations ===================
    if 'location_countries' in d.keys():
        new_d['location_countries'] = [country2continent[country] for country in new_d['location_countries']['country']]

    # =================== Process created (date) ===================
    if 'start_date' in d.keys():
        years = trial_date2float(d['start_date'][0])
    else: # 'study_first_submitted' in d.keys():
        years = trial_date2float(d['study_first_submitted'][0])
    new_d['start_date'] = years

    # # =================== Process enddate (date) ===================
    if 'primary_completion_date' in d.keys():
        years = trial_date2float(d['primary_completion_date'][0])
    elif 'completion_date' in d.keys():
        years = trial_date2float(d['completion_date'][0])        
    else: #'last_update_posted' in d.keys():
        years = trial_date2float(d['last_update_posted'][0])
    new_d['completion_date'] = years

    # =================== Process intervention ===================
    new_d['intervention_types'] = [_['intervention_type'][0] for _ in d['intervention']] if 'intervention' in d.keys() else []
    new_d['intervention_names'] = [_['intervention_name'][0] for _ in d['intervention']] if 'intervention' in d.keys() else []

    if 'Drug' not in new_d['intervention_types'] and 'Biological' not in new_d['intervention_types']: 
    # if 'Biological' not in new_d['intervention_types']: 
        return None, 'non-Biological or Drug' 


    # =================== Process conditions ===================
    new_d['conditions'] = []
    new_d['icdcode_lst'] = []
    new_d['conditions_not_found'] = []        

    if 'condition' in d.keys():
        new_d['conditions'] = [i.lower() for i in d['condition']]
        new_d['icdcode_lst'] = []
        new_d['conditions_not_found'] = []
        for disease in new_d['conditions']:
            if disease in disease2icd.keys():
                new_d['icdcode_lst'].append(disease2icd[disease])
            else:
                new_d['conditions_not_found'].append(disease)

    if len(new_d['icdcode_lst']) == 0:
        new_d['icdcode_lst'] = [['None']]
    
    # =================== Process label ===================
    new_d['label'] = nctid2label[new_d['id_info']] if new_d['id_info'] in nctid2label.keys() else -1
    why_stop = new_d['why_stopped'] if 'why_stopped' in d.keys() else 'None'

    if (new_d['label'] == -1) and ('lack of efficacy' in why_stop or 'efficacy concern' in why_stop or 'accrual' in why_stop):
        new_d['label'] = 0
    
    if new_d['label'] == -1:
        return None, 'no-label'

    # =================== Process Criteria ===================
    criteria_dict = d['eligibility'][0] if 'eligibility' in d.keys() else {}
    if 'criteria' not in criteria_dict.keys():
        new_d['eligibility_text'] = 'None'
    else:
        new_d['eligibility_text'] = criteria_dict['criteria'][0]['textblock'][0] if 'textblock' in criteria_dict['criteria'][0].keys() else 'None'
    new_d['eligibility_text'] = clean_protocol(new_d['eligibility_text'])
    new_d['eligibility_gender'] = criteria_dict['gender'][0] if 'gender' in criteria_dict.keys() else 'None'
    new_d['eligibility_min_age'] = criteria_dict['minimum_age'][0] if 'minimum_age' in criteria_dict.keys() else 'N/A'
    new_d['eligibility_max_age'] = criteria_dict['maximum_age'][0] if 'maximum_age' in criteria_dict.keys() else 'N/A'

    new_d['eligibility_min_age'] = process_age(new_d['eligibility_min_age'])
    new_d['eligibility_max_age'] = process_age(new_d['eligibility_max_age'])

    # =================== Process study_design_info ===================
    design_dict = d['study_design_info'][0] if 'study_design_info' in d.keys() else {}

    new_d['allocation'] = design_dict['allocation'][0] if 'allocation' in design_dict.keys() else 'N/A'
    new_d['intervention_model'] = design_dict['intervention_model'][0] if 'intervention_model' in design_dict.keys() else 'None'
    new_d['primary_purpose'] = design_dict['primary_purpose'][0] if 'primary_purpose' in design_dict.keys() else 'None'
    new_d['masking'] = design_dict['masking'][0] if 'masking' in design_dict.keys() else 'None (Open Label)'

    # =================== Process sponsors ===================
    top10 = ['GlaxoSmithKline', 'Merck Sharp & Dohme LLC', 'Sanofi Pasteur, a Sanofi Company', 
             'Amgen', 'Pfizer', 'National Cancer Institute (NCI)', 'Novartis Pharmaceuticals', 
             'Abbott', 'Bristol-Myers Squibb', 'Novartis Vaccines']
    sponsor = d['sponsors'][0]['lead_sponsor'][0]['agency'][0]
    new_d['sponsors'] = "Large" if sponsor in top10 else "Small"

    # =================== summarize additional info ===================
    new_d['brief_summary_additional'] = []
    for feature_name in ['eligibility_gender', 'eligibility_min_age', 'eligibility_max_age', 
        'allocation', 'intervention_model', 'primary_purpose', 'masking', 'sponsors', 'location_countries']:

        feature = new_d[feature_name]
        if feature_name == "location_countries":
            feature = ','.join(np.unique(feature))
        new_d['brief_summary_additional'].append(feature_name.replace('_',' ')+': '+feature.lower())
        
    return new_d, 'success'

def get_drug_mapping(all_drug_dict):
    drug_mapping = {}
    for main_drug in all_drug_dict.keys():
        for synonym in all_drug_dict[main_drug]['synonyms']:
            drug_mapping[synonym.lower()] = main_drug
        for synonym in all_drug_dict[main_drug]['international-brands']:
            drug_mapping[synonym.lower()] = main_drug
        for synonym in all_drug_dict[main_drug]['products']:
            drug_mapping[synonym.lower()] = main_drug
    return drug_mapping


def get_drugs_from_trial(trials, drug_mapping, all_drug_dict):
    drugs_found = []
    for trial in trials:
        # print(trial['intervention_types'], trial['intervention_names'])

        trial_drugs = [trial['intervention_names'][i]
            for i in range(len(trial['intervention_names'])) 
            if trial['intervention_types'][i] in ('Biological','Drug')]
        
        drugs_found_ = []
        for drug in trial_drugs:
            if drug.lower() in drug_mapping.keys():
                drugs_found_.append(all_drug_dict[drug_mapping[drug.lower()]])
            else: # append None
                drugs_found_.append({'description':drug.lower(), 'pharmacodynamics':'None', 'toxicity':'None', 'metabolism':'None', 'absorption':'None'})
        drugs_found.append(drugs_found_)
        
    return drugs_found

if __name__=='__main__':
    ## ========== Process Drugs and Biologics ==========
    with open('../data/drugbank_full_database.xml') as xml_file:
        tree = xml.etree.ElementTree.parse(xml_file)
    root = tree.getroot()

    all_drug_dict = {}
    for i, child in enumerate(tqdm(root)):
        d = dictify(child, root=False, str_to_replace='{http://www.drugbank.ca}')
        d = parse_drug_dict(d)
        all_drug_dict[d['name']] = d

    pickle.dump(all_drug_dict, open('all_drug_dict2.pkl', 'wb'))

    ## ========== Process Clinical Trials ==========
    def process_trial(args): 
        file, nctid2label, disease2icd, country2continent = args

        with open(file, 'r') as xml_file:
            tree = xml.etree.ElementTree.parse(xml_file)
        root = tree.getroot()

        d = dictify(root)
        trial, flag = parse_trial_dict(d, nctid2label=nctid2label, disease2icd=disease2icd, country2continent=country2continent)
        return trial, flag

    with open("../data/all_xml", 'r') as fin:
        lines = fin.readlines()
    input_file_lst = [i.strip() for i in lines]

    nctid2label = nctid2label_dict2(outcome2label_file="../IQVIA/outcome2label.txt", nctid2outcome_file="../IQVIA/trial_outcomes_v1.csv")
    disease2icd = load_disease2icd2(disease_file='../data/diseases.csv')
    country2continent = load_country2continent(countries_file='../data/countries.csv')

    parent_path = '../'

    with multiprocessing.Pool(processes=20) as pool: # parallelize for speed, 426368 total trials
        paths = [parent_path+f for f in input_file_lst]
        nctid2label_list = [nctid2label]*len(paths)
        disease2icd_list = [disease2icd]*len(paths)
        country2continent_list = [country2continent]*len(paths)

        out = pool.map(process_trial, zip(paths, nctid2label_list, disease2icd_list, country2continent_list))
        all_flags = [_[1] for _ in out]
        all_trials = [_[0] for _ in out]

    pickle.dump([all_trials, all_flags], open('all_trials_and_flags.pkl', 'wb'))
    all_valid_trials = [all_trials[i] for i in range(len(all_trials)) if all_flags[i]=='success']
    print(np.unique(all_flags, return_counts=True))
    # pickle.dump(all_valid_trials, open('all_valid_trials2.pkl', 'wb'))

    ## ========== OBTAIN PRETRAINED EMBEDDINGS ==========
    # all_drug_dict = pickle.load(open('all_drug_dict2.pkl', 'rb'))
    # all_valid_trials = pickle.load(open('all_valid_trials2.pkl', 'rb'))
    print('len all valid files:', len(all_valid_trials))

    drug_mapping = get_drug_mapping(all_drug_dict=all_drug_dict)
    all_valid_drugs = get_drugs_from_trial(all_valid_trials, drug_mapping=drug_mapping, all_drug_dict=all_drug_dict)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_model = SentenceTransformer("dmis-lab/biobert-base-cased-v1.2", device=device)
    for i in tqdm(range(len(all_valid_trials))):
        all_text = [all_valid_trials[i]['brief_summary']] + all_valid_trials[i]['brief_summary_additional'] + all_valid_trials[i]['eligibility_text']
        for drug in all_valid_drugs[i]:
            all_text.extend([drug['description'], drug['pharmacodynamics'], drug['toxicity'], drug['metabolism'], drug['absorption']])
        all_text_embeds = base_model.encode(all_text, convert_to_tensor=True)
        all_valid_trials[i]['all_text_embeds'] = all_text_embeds.cpu()

    pickle.dump(all_valid_trials, open('all_valid_trials2.pkl', 'wb'))


