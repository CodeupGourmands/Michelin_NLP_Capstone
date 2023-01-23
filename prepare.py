import pandas as pd
import numpy as np


def clean_michelin(df):
    '''
    This function was the basic clean used before scraping the data
    into the new data frame
    '''
    # Make all column names lowercase
    df.columns = df.columns.str.lower()
    # renaming columns in snake case
    df = df.rename(columns = {"phonenumber":"phone_number",
                              "websiteurl":"website_url",
                              "facilitiesandservices":"facilities_and_services"})
    return df
