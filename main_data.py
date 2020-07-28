# -*- coding: utf-8 -*-
"""
Created on Sun May 10 08:49:19 2020

@author: philv
"""

import os
from covid_scraper import MetaCollector
import datetime
import sys
import pandas as pd
import time
import numpy as np

if __name__ == "__main__":
    platform = 'medrxiv'
   
    ''' This will collect metadata on all submissions that were last updated 
    at the starting date or later. This means that the initial submission
    might be earlier than the starting date.
    '''
   
        
    MetaCollector.collect_data(platform) 
    MetaCollector.collect_data_status(platform)  
    
    '''
    Tag the submissions as COVID related based on a regex match in the title
    and in the abstract.
    '''
    regex_search = (
                    "(\\s|\\b)(ncov)([^a-z]|\\b)|"
                    "(\\s|\\b)(novel)[\\s-]?(corona\\s*virus)([^a-z]|\\b)|"
                    "(\\s|\\b)(sars-cov-2)([^a-z]|\\b)|"
                    "(\\s|\\b)(covid)([^a-z]|\\b)|"
                    "(\\s|\\b)(corona\\s*virus)[\\s-]*?2(\\s|\\b)|"
                    "(\\s|\\b)(corona\\s*virus disease 2019)([^a-z]|\\b)"
                    )
    MetaCollector.tag_keywords_title(platform,regex_search)   

        
    

