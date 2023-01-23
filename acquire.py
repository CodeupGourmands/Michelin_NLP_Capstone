import pandas as pd
import numpy as np
import re
import prepare as p

# Webscraping
import requests
from requests import get
from bs4 import BeautifulSoup
import time


def get_michelin_pages():
    '''
    This function takes the original kaggle dataset, applies a cleaning
    function, and then scrapes the michelin website for all review text,
    and appends the dataframe with the review text for the specific restaurant
    row-wise. The review text is under a new column "data"
    '''
    df = pd.read_csv('data/michelin_my_maps.csv')
    df = p.clean_michelin(df)
    urls = df['url']
    output = []
    for url in urls:
        try:
            request = requests.get(url)
            soup = BeautifulSoup(request.content, 'html.parser')
            page_output = soup.find('div',
                                    class_='restaurant-details__description--text').find('p').text.strip()
            text = page_output
            output.append(page_output)
            time.sleep(2)
        except AttributeError:
            page_output = 'None'
            output.append(page_output)
            time.sleep(5)
            continue
    df['data'] = output
    return df

