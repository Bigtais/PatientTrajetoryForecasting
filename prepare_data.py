#!/usr/bin/env python
# coding: utf-8


# In[28]:


from utils.data_preparation import (
    load_mimic_data,
    clean_data,
    icd_mapping,
    trim,
    build_data,
    remove_code,
    save_files,
    generate_code_types,
    filter_subjects,
    filter_notes
)

from utils.utils import get_paths, store_files
import yaml 
import os
import pandas as pd
import asyncio


# In[ ]:


with open('new_paths.yaml', 'r') as file:
    path_config = yaml.safe_load(file)


# In[5]:


paths = get_paths(path_config) 


# In[6]:

async def await_outside_function():
    subject_id_adm_map, adm_dx_map, adm_px_map, adm_drug_map, drug_description_map, notes = await load_mimic_data(**paths, load_notes = False)
    return subject_id_adm_map, adm_dx_map, adm_px_map, adm_drug_map, drug_description_map, notes


subject_id_adm_map, adm_dx_map, adm_px_map, adm_drug_map, drug_description_map, notes = asyncio.run(await_outside_function())


# In[7]:


NUM_VISITS_PLOT = 15
MIN_VISITS = 2


# In[9]:


subject_id_adm_map, adm_dx_map, adm_px_map, adm_drug_map = clean_data(subject_id_adm_map, adm_dx_map, adm_px_map, adm_drug_map, MIN_VISITS)


# In[11]:


adm_dx_map, adm_px_map, code_description_map, _, __ = icd_mapping(adDx = adm_dx_map, adPx = adm_px_map, adDrug = adm_drug_map, drugDescription = drug_description_map, **paths)


# In[12]:


max_dx, max_px, max_drg = 80, 80, 80 
adm_dx_map, adm_px_map, adm_drug_map = trim(adm_dx_map, adm_px_map, adm_drug_map, max_dx, max_px, max_drg)


# In[13]:


patients_visits_sequences, tokens_to_ids_map = build_data(subject_id_adm_map, adm_dx_map, adm_px_map, adm_drug_map)


# In[14]:


min_freq_codes = 5
patients_visits_sequences, tokens_ids_map, ids_tokens_map  = remove_code(patients_visits_sequences, tokens_ids_map, threshold = min_freq_codes)


# In[29]:


save_files(patients_visits_sequences, dict(tokens_ids_map), code_description_map)


# In[ ]:


code_types = generate_code_types(ids_tokens_map)

