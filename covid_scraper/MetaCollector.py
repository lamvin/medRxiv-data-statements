# -*- coding: utf-8 -*-
"""
Created on Sun May 10 08:55:58 2020

@author: philv
"""
import os
import csv
import datetime
from bs4 import BeautifulSoup as bs
import cfscrape
import time
import numpy as np

import logging
import requests
import xml.etree.cElementTree as ET
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import pandas as pd
import dateutil

    
def find_last_day_collect(platform):
    dates = set()
    with open(os.path.join("data","meta",platform + '.csv'),'r',encoding='utf-8') as f:
        reader = csv.reader(f,delimiter='|')
        for line in reader:
            dates.add(line[1])
   
    dates = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in dates]
    most_recent = max(dates)
    return most_recent

def collect_data(platform):
    start_date = find_last_day_collect(platform)
    collect_MB(platform,start_date)
   
       
def date_range(date1, date2):
    dates = []
    for n in range(int ((date2 - date1).days)+1):
        dates.append(date1 + datetime.timedelta(n))
    return dates

#Function to collect data from bioRxiv and medRxiv
def collect_MB(platform,start_date):
    scraper = cfscrape.create_scraper() # returns a requests.Session object
    url_base = "http://{0}.org/search/jcode%3A{0}%20limit_from%3A{1}%20limit_to%3A{1}%20numresults%3A100%20format_result%3Astandard?page={2}"
    now = datetime.datetime.now()
    dates_to_collect = date_range(start_date,now)
    nb_dates = len(dates_to_collect)
    with open(os.path.join('data','meta',platform+'.csv'),'a',encoding='utf-8') as f:
        for i in range(nb_dates):
            dt = dates_to_collect[i].strftime("%Y-%m-%d")
            print('{}: collecting metadata {}, {}/{}.'.format(platform,dt,i+1,nb_dates))
            page=0
            while True:
                time.sleep(5) 
                url_search = url_base.format(platform,dt,page)
                attempts = 0
                max_attempts = 3
                while True:
                    try:
                        html = bs(scraper.get(url_search).text,"html.parser")
                        break
                    except:
                        if attempts == 3:
                            raise Exception('No response from host.')
                        print('Host not responding, attempt {}/{}.'.format(attempts,max_attempts))
                        time.sleep(60)
                        attempts += 1
                # Collect the articles in the result in a list
                articles = html.find_all('li', attrs={'class': 'search-result'})
                if len(articles) == 0:
                    break
                if page == 0:
                    temp = html.find('section',attrs={'id':'section-content'}).find('div', attrs={'class':"pane-content"}).text
                    temp = temp.strip()
                    nb_results = int(temp.split(' ')[0])
                    nb_pages = int(np.ceil(nb_results/100))
                
                for j in range(len(articles)):
                    article = articles[j]
                    try:
                        art_doi = article.find('span', attrs={'class': "highwire-cite-metadata-doi highwire-cite-metadata"})
                        art_doi = ':'.join(art_doi.text.split(':')[1:]).strip()
                        # Pull the title, if it's empty then skip it
                        title = article.find('span', attrs={'class': 'highwire-cite-title'})
                        if title is None:
                            continue
                        title = title.text.strip()
                        
                        
                        # Now collect author information
                        authors = article.find_all('span', attrs={'class': 'highwire-citation-author'})
                        all_authors = []
                        for author in authors:
                            name = author.text
                            name = name.split(' ')
                            name = '/'.join([name[-1]] + [' '.join(name[:-1])])
                            all_authors.append(name)
                        all_authors = ';'.join(all_authors)
                        items = [art_doi,dt,title,all_authors]
                        items = [x.replace('|',' ') for x in items]
                        items = [x.replace('\n',' ') for x in items]
                        f.write('|'.join(items)+'\n')  
                    except AttributeError:
                        continue
                if page == nb_pages - 1:
                    break
                else:
                    page += 1
 
    
def tag_keywords_title(platform,regex_search):
    meta_data = pd.read_csv(os.path.join("data","meta",platform+".csv"),
                            sep="|",header=None,error_bad_lines=False)
    meta_data.columns = ["ID","date","title","authors"]
    meta_data = meta_data.loc[~meta_data['ID'].isnull()]
    meta_data['title'] = meta_data['title'].str.lower()
    meta_data.loc[meta_data['title'].isnull(),"title"] = ''
    meta_data['covid_related'] = meta_data['title'].apply(lambda x: re.search(regex_search,x) is not None)
    meta_data[['ID','date','covid_related','title']].to_csv(os.path.join("data","meta",platform+"_key.csv"),index=False,sep='|')
 