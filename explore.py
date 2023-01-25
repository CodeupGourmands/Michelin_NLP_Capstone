from typing import Union
import numpy as np
import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def get_ngram_frequency(ser: pd.Series, n: int = 1) -> pd.Series:
    # TODO Docstring
    words = ' '.join(ser).split()
    if n > 1:
        ngrams = nltk.ngrams(words, n)
        words = [' '.join(n) for n in ngrams]
    return pd.Series(words).value_counts()


def generate_word_cloud(ser: pd.Series, ngram: int = 1,
                        ax: Union[plt.Axes, None] = None,
                        **kwargs) -> Union[plt.Axes, None]:
    # TODO Docstring
    if ser.dtype != np.int64:
        ser = get_ngram_frequency(ser, ngram)
    wc = WordCloud(**kwargs).generate_from_frequencies(ser.to_dict())
    if ax is not None:
        ax.imshow(wc)
        return ax
    plt.imshow(wc)
    plt.show()

def get_award_freq(train):
    '''
    This function takes in the training data set and creates a countplot
    utilizing Seaborn to visualize the range and values of award
    categories in the training dataset'''
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(figsize=(9, 6))
    cpt = sns.countplot(x='award',
                        data=train,
                        palette='RdYlGn_r',
                        order = train['award'].value_counts().index)
    plt.title('Bib Gourmand is the Most Common Award Level in our Dataset')
    plt.xlabel("Award Level")
    plt.ylabel('Count of Restaurants')
    for tick in axes.xaxis.get_major_ticks():
        tick.label1.set_fontsize(10)
    plt.show()