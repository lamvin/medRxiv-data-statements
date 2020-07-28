# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:22:57 2020

@author: philv
"""
import pandas as pd
import os
import re
import numpy as np

patterns = {"conditional_availability":"request|upon|author|on demand",
            "publicly_available":"(are|were|is) collected|\\bpublic(ly)?\\b|open.?source|available.?online|"+
                "public source",
            "not_applicable":"not ap.?licable|\\bn.?a\\b|no\\s([a-zA-Z]*\\s){0,1}(data|code)\\s([a-zA-Z]*\\s){0,1}(use|referred)",
            "not_available":"\\b(no|not|none).*(available|online|share)",
            "is_available":"(is|are) available",
            "future":"will be(\\b.*\\b)?available",
            "hyperlink":"https",
            "within_manuscript":"supplement|(included|available) (with)?in\\s([a-zA-Z]*\\s){0,1}(main text|article|manuscript)" 
            }

types = ["code","data"]


def find_pattern_sentence(text,pattern,type_mat):
    sentences = re.split('[\\.]',text)
    match = False
    for sent in sentences:
        if re.search(patterns[pattern],sent):
            if pattern == "not_applicable":
                match = True
                break
            if re.search(type_mat,sent):
                match = True
                break
    return match

df_title = pd.read_csv(os.path.join("data","meta","medrxiv_key.csv"),sep='|')
df_status = pd.read_csv(os.path.join("data","meta","medrxiv_data_status.csv"),sep='|',header=None)
df_status.columns = ['ID',"url_data",'statement']

df = pd.merge(df_title,df_status,how="outer",on="ID")
df.loc[df['statement'].isnull(),"statement"] = ""
df["statement"] = df["statement"].str.lower() 
df["statement"] = df["statement"].apply(lambda x : x.replace('dr.','dr'))
    
for type_mat in types:
    for pattern in patterns:
        df[pattern+'_'+type_mat] = df['statement'].apply(lambda x: find_pattern_sentence(x,pattern,type_mat))
        
levels = ["conditional_availability",
             "not_available",
     "future",
    "publicly_available",
    "not_applicable",
    "within_manuscript",
    "is_available"]

nb_levels = len(levels)    

for type_mat in types:
    for i in range(nb_levels):
        level = levels[i]
        priority = levels[:i]
        level_val = df[level+'_'+type_mat]
        priority_vals = ~df[[x+'_'+type_mat for x in priority]].any(axis=1)
        df[level+'_'+type_mat] = priority_vals & level_val

df.to_excel(os.path.join("data","meta","medrxiv_tagged.xlsx"))
        

vars_code = {x+'_code':['mean','sum'] for x in patterns.keys()}        
vars_data = {x+'_data':['mean','sum'] for x in patterns.keys()}

df['date'] = pd.to_datetime(df['date'])
stats_date = df.groupby([pd.Grouper(freq='Y',key='date',label='left'),pd.Grouper('covid_related')]).agg({'ID':'count'})
df = df[df['date'].between('2020-01-01', '2020-12-30')]
stats_code = df.groupby(['covid_related']).agg(vars_code)        
stats_data = df.groupby(['covid_related']).agg(vars_data)     

nb_articles_covid = df['covid_related'].sum()


#Make figures 
from matplotlib import pyplot as plt

plt.figure(figsize=(12,12))
plt.subplot(1,2,1)
stats = stats_data
grps = stats.columns
grps = ['_'.join(x.split('_')[:-1]) for x in grps]
grp_labels = np.array(['Conditional availability', 'Publicly available',
              'Not applicable','Not available','Available',
              'Available later','Hyperlink','Within manuscript'])

order = np.array([2,3,6,0,5,7,1,4])
idx = np.arange(len(grps))
vals_cov = stats.loc[True]
vals_no_cov = stats.loc[False]
bar_width=0.4
rects = plt.barh(idx, vals_cov[order]*100, bar_width,
       alpha=0.5,
       color='r',
       label= 'COVID related')
rects2 = plt.barh(idx+bar_width, vals_no_cov[order]*100, bar_width,
       alpha=0.5,
       color='b',
       label= 'Others')

plt.yticks(idx + (bar_width)*0.5, grp_labels[order])
plt.xlabel('% of manuscripts')
plt.legend()
plt.title('Data',fontsize=20)
plt.tick_params(labelsize=15)
plt.rcParams.update({'font.size': 15})

plt.subplot(1,2,2)
stats = stats_code
grps = stats.columns
idx = np.arange(len(grps))
vals_cov = stats.loc[False]
vals_no_cov = stats.loc[True]
bar_width=0.4
rects = plt.barh(idx, vals_cov[order]*100, bar_width,
       alpha=0.5,
       color='r',
       label= 'COVID related')
rects2 = plt.barh(idx+bar_width, vals_no_cov[order]*100, bar_width,
       alpha=0.5,
       color='b',
       label= 'Others')
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    labelleft=False) # labels along the bottom edge are off
plt.title('Code',fontsize=20)
plt.xlabel('% of manuscripts')
plt.tick_params(labelsize=15)
plt.rcParams.update({'font.size': 15})
plt.savefig('data_statements.svg')
plt.savefig('data_statements.png')

