import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union,Tuple
import acquire as a
import prepare as p
# Custom function to create facilities DataFrame split
def prepare_facilities(df: pd.DataFrame,
                       split: bool = True) -> Union[pd.DataFrame,
                                                    Tuple[pd.DataFrame,
                                                          pd.DataFrame,
                                                          pd.DataFrame]]:
    '''
    #### Description:
    Prepares Michelin DataFrame

    #### Required Imports:
    from typing import List, Union, Tuple
    import pandas as pd
    import prepare as p

    #### Parameters:
    df: `DataFrame` with Michelin data
    split: Boolean for whether or not to split the data, default True

    #### Returns:
    either a DataFrame or a tuple of the Train, Validate, and Test
    `DataFrame`
    '''

    df = p.create_features(df)
    df = p.change_dtype_str(df)
    df = pd.concat([df, p.process_nl(df.facilities_and_services)], axis=1)
    df['word_count'] = df.lemmatized.str.split().apply(len)
    if split:
        return p.tvt_split(df, stratify='award')
    return df

    ############################
##### Global Variables #####
############################

# Get the data
df = a.get_michelin_pages()

# Splitting our data (56% Train, 24% Validate, 20% Test)
train, validate, test = p.prepare_michelin(df)

# Run Facilities split
f_train, f_validate, f_test = prepare_facilities(df)
train.head(2)

# Assign all, 1_star, 2_star, 3_star and bib_gourmand reviews by passing the function with a join
all_reviews = (' '.join(train['lemmatized']))
one_star_reviews = (
    ' '.join(train[train.award == '1 michelin star']['lemmatized']))
two_star_reviews = (
    ' '.join(train[train.award == '2 michelin stars']['lemmatized']))
three_star_reviews = (
    ' '.join(train[train.award == '3 michelin stars']['lemmatized']))
bib_gourmand_reviews = (
    ' '.join(train[train.award == 'bib gourmand']['lemmatized']))

# Break them all into word lists with split
all_reviews_words = all_reviews.split()
one_star_reviews_words = one_star_reviews.split()
two_star_reviews_words = two_star_reviews.split()
three_star_reviews_words = three_star_reviews.split()
bib_gourmand_reviews_words = bib_gourmand_reviews.split()

# Assign word counts to Frequency Variables
freq_one_star_reviews = pd.Series(one_star_reviews_words).value_counts()
freq_two_star_reviews = pd.Series(two_star_reviews_words).value_counts()
freq_three_star_reviews = pd.Series(
    three_star_reviews_words).value_counts()
freq_bib_gourmand_reviews = pd.Series(
    bib_gourmand_reviews_words).value_counts()
freq_all_reviews = pd.Series(all_reviews_words).value_counts()

## --------------------------- ##
## CREATE facilities variables ##
## --------------------------- ##

# Assign all, 1_star, 2_star, 3_star and bib_gourmand lists by passing the clean function with a join
all_facilities = ' '.join(f_train['lemmatized'])
one_star_facilities = ' '.join(
    f_train[f_train.award == '1 michelin star']['lemmatized'])
two_star_facilities = ' '.join(
    f_train[f_train.award == '2 michelin stars']['lemmatized'])
three_star_facilities = ' '.join(
    f_train[f_train.award == '3 michelin stars']['lemmatized'])
bib_gourmand_facilities = ' '.join(
    f_train[f_train.award == 'bib gourmand']['lemmatized'])

# Break them all into word lists with split
all_facilities_words = all_facilities.split()
one_star_facilities_words = one_star_facilities.split()
two_star_facilities_words = two_star_facilities.split()
three_star_facilities_words = three_star_facilities.split()
bib_gourmand_facilities_words = bib_gourmand_facilities.split()

# Assign word counts to Frequency Variables
freq_one_star_facilities = pd.Series(
    one_star_facilities_words).value_counts()
freq_two_star_facilities = pd.Series(
    two_star_facilities_words).value_counts()
freq_three_star_facilities = pd.Series(
    three_star_facilities_words).value_counts()
freq_bib_gourmand_facilities = pd.Series(
    bib_gourmand_facilities_words).value_counts()
freq_all_facilities = pd.Series(all_facilities_words).value_counts()

## -------------------------- ##
## Create Frequency DataFrame ##
## -------------------------- ##

word_counts_df = pd.concat([freq_all_facilities,
                            freq_one_star_facilities,
                            freq_two_star_facilities,
                            freq_three_star_facilities,
                            freq_bib_gourmand_facilities,
                            freq_all_reviews,
                            freq_one_star_reviews,
                            freq_two_star_reviews,
                            freq_three_star_reviews,
                            freq_bib_gourmand_reviews], axis=1
                           ).fillna(0).astype(int)

word_counts_df.columns = ['all_facilities',
                          'one_star_facilities',
                          'two_star_facilities',
                          'three_star_facilities',
                          'bib_gourmand_facilities',
                          'all_reviews',
                          'one_star_reviews',
                          'two_star_reviews',
                          'three_star_reviews',
                          'bib_gourmand_reviews']

word_counts_df

# Create word_count variables
facilities_wc_by_award = f_train.groupby('award').word_count.mean()
reviews_wc_by_award = train.groupby('award').word_count.mean()
