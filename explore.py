import prepare as p
from typing import List, Union, Tuple
from typing import Union
import numpy as np
import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import scipy.stats as stats
from scipy.stats import ttest_ind, levene, f_oneway


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
                        order=train['award'].value_counts().index)
    plt.title('Bib Gourmand is the Most Common Award Level in our Dataset')
    plt.xlabel("Award Level")
    plt.ylabel('Count of Restaurants')
    for tick in axes.xaxis.get_major_ticks():
        tick.label1.set_fontsize(10)
    plt.show()


def get_wordcount_bar(train):
    '''
    This function takes in the training dataset and creates a bar plot of the
    average wordcount of a review based on the Michelin Star Award
    '''
    # Use groupby to get an average length per language
    review_wordcount = train.groupby(
        'award').word_count.mean().sort_values(ascending=False)
    # Set style, make a chart
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(figsize=(9, 6))
    ax = sns.barplot(x=review_wordcount.values,
                     y=review_wordcount.index, palette='coolwarm')
    plt.title('Average Wordcount of Michelin Star Level Restaurants')
    plt.xlabel("Average Word Count")
    plt.ylabel('Award Level')
    plt.show()


def top_10_country_viz(train):
    '''
    This function takes in the training dataset and creates a bar plot of the
    top 10 countries with Michelin restaurants
    '''
    # Use groupby to get an average length per language
    top_10_countries = train['country'].value_counts().head(10)
    # Set style, make a chart
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(figsize=(9, 6))
    ax = sns.barplot(x=top_10_countries.index,
                     y=top_10_countries.values,
                     palette='mako')
    plt.title('Countries with the Most Michelin Restaurants')
    plt.xlabel("Countries")
    plt.ylabel('Number of Restaurants')
    plt.show()


def sentiment_scores_bar(train):
    dfg = train.groupby(['award'])[
        'sentiment'].mean().sort_values(ascending=False)
    # create a bar plot
    dfg.plot(kind='bar', title='Sentiment Score', ylabel='Mean Sentiment Score',
             xlabel='', fontsize=20, color=['#beaed4', '#f0027f', '#7fc97f', '#fdc086'])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=22)
    plt.show()

# -----------------------------Stats Tests-------------------------------#


def get_anova_wordcount(train):
    '''
    This function creates separate dataframes for
    each award category, and utilizes an ANOVA test
    to compare the mean word count of each category. It
    returns the test statistic, p-value, and treatment
    of the null hypothesis
    '''
    # Create separate df for each category
    train_bib = train[train.award == 'bib gourmand']
    train_onestar = train[train.award == '1 michelin star']
    train_twostar = train[train.award == '2 michelin stars']
    train_threestar = train[train.award == '3 michelin stars']
    # set alpha
    alpha = 0.05
    # Run the test
    f, p = stats.f_oneway(train_bib.word_count, train_onestar.word_count,
                          train_twostar.word_count, train_threestar.word_count)
    if p < alpha:
        print('We reject the null hypothesis. There is sufficient\n'
              'evidence to conclude that the word count is significantly\n'
              'different between award categories.')
    else:
        print("We fail to reject the null hypothesis.")
    return print(f'Test Statistic: {f}, P Statistic: {p}')


def get_stats_ttest(df):
    '''Function returns statistical T test'''
    Two_Star = df[df.award == '2 michelin stars']
    Three_Star = df[df.award == '3 michelin stars']
    stat, pval = stats.levene(Two_Star.sentiment,
                              Three_Star.sentiment)
    alpha = 0.05
    if pval < alpha:
        variance = False

    else:
        variance = True
    t_stat, p_val = stats.ttest_ind(Two_Star.sentiment,
                                    Three_Star.sentiment,
                                    equal_var=True, random_state=123)

    print(f't_stat= {t_stat}, p_value= {p_val/2}')
    print('-----------------------------------------------------------')
    if (p_val/2) < alpha:
        print(f'We reject the null Hypothesis')
    else:
        print(f'We fail to reject the null Hypothesis. There is no significant difference between the sentiment scores.')


###---------------------------------WordClouds--------------------------------###

def get_threestar_wordcloud():
    '''
    This function utilizes a text file of all three-star review
    words and a pre-selected image to create a word cloud containing
    all three star words in an image cloud format. It takes the text
    file and image file from the /images folder.
    '''
    #Import TXT file of all three star words
    threestar_text = open(
            "./images/all_threestar_words.txt",
            mode='r', encoding='utf-8').read()
    #Import .png file of three star logo, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/three_stars.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors
    colors = ['purple', 'gold']
    custom_cmap = mcolors.ListedColormap(colors)
    #Make the wordcloud, generate the image
    wc = WordCloud(
               mask = mask, background_color = "black",
               max_words = 400, max_font_size = 500,
               random_state = 42, width = mask.shape[1],
               colormap= custom_cmap,
               contour_color='gold', contour_width=2,
               height = mask.shape[0])
    wc.generate(threestar_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()



def get_twostar_wordcloud():
    '''
    This function utilizes a text file of all two-star review
    words and a pre-selected image to create a word cloud containing
    all two star words in an image cloud format. It takes the text
    file and image file from the /images folder.
    '''
    #Import TXT file of all two star words
    twostar_text = open("./images/all_twostar_words.txt",
            mode='r', encoding='utf-8').read()
    #Import .png file of three star logo, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/two_stars.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors
    colors = ['blue', 'red']
    custom_cmap = mcolors.ListedColormap(colors)
    #Make the wordcloud, generate the image
    wc = WordCloud(
               mask = mask, background_color = "lightyellow",
               max_words = 1000, max_font_size = 500,
               random_state = 42, width = mask.shape[1],
               colormap= custom_cmap,
               contour_color='red', contour_width=1,
               height = mask.shape[0])
    wc.generate(twostar_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()



def get_onestar_wordcloud():
    '''
    This function utilizes a text file of all one-star review
    words and a pre-selected image to create a word cloud containing
    all one star words in an image cloud format. It takes the text
    file and image file from the /images folder.
    '''
    #Import TXT file of all one star words
    onestar_text = open("./images/all_onestar_words.txt",
            mode='r', encoding='utf-8').read()
    #Import .png file of three star logo, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/one_star_heart.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors
    colors = ['firebrick', 'orangered']
    custom_cmap = mcolors.ListedColormap(colors)
    #Make the wordcloud, generate the image
    wc = WordCloud(
               mask = mask, background_color = "lightgray",
               max_words = 250, max_font_size = 500,
               random_state = 42, width = mask.shape[1],
               colormap= custom_cmap,
               contour_color='crimson', contour_width=1.5,
               height = mask.shape[0])
    wc.generate(onestar_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()



# Bib Gourmand Word Cloud

def get_bib_wordcloud():
    '''
    This function utilizes a text file of all bib gourmand review
    words and a pre-selected image to create a word cloud containing
    all bib gourmand words in an image cloud format. It takes the text
    file and image file from the /images folder.
    '''
    #Import TXT file of all bib gourmand star words
    bib_text = open("./images/all_bib_words.txt",
            mode='r', encoding='utf-8').read()
    #Import .png file of bib gourmand image, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/bib_gourmand.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors
    colors = ['darkred', 'orangered']
    custom_cmap = mcolors.ListedColormap(colors)
    #Make the wordcloud, generate the image
    wc = WordCloud(
               mask = mask, background_color = "white",
               max_words = 500, max_font_size = 500,
               random_state = 42, width = mask.shape[1],
               colormap= custom_cmap,
               contour_color='maroon', contour_width=1.5,
               height = mask.shape[0])
    wc.generate(bib_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()



#########################
##### Justin's Code #####
#########################


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
import acquire as a
import prepare as p

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

data= [freq_all_facilities,
                        freq_one_star_facilities,
                        freq_two_star_facilities,
                        freq_three_star_facilities,
                        freq_bib_gourmand_facilities,
                        freq_all_reviews,
                        freq_one_star_reviews,
                        freq_two_star_reviews,
                        freq_three_star_reviews,
                        freq_bib_gourmand_reviews]
cols = ['all_facilities',
                       'one_star_facilities',
                       'two_star_facilities',
                       'three_star_facilities',
                       'bib_gourmand_facilities',
                       'all_reviews',
                       'one_star_reviews',
                       'two_star_reviews',
                       'three_star_reviews',
                       'bib_gourmand_reviews']

word_counts = pd.DataFrame(data, cols)

"""
word_counts = pd.concat(, axis=1
                        ).fillna(0).astype(int)

word_counts.columns = 
"""
# Create word_count variables
facilities_wc_by_award = f_train.groupby('award').word_count.mean()
reviews_wc_by_award = train.groupby('award').word_count.mean()


##########################
##### Visualizations #####
##########################

def QMCBT_viz_wc():
    img = WordCloud(background_color='white'
                ).generate(' '.join(all_reviews_words))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Most Common Review Words')
    return plt.show()

def QMCBT_viz_1():

    # Plot Top-20 Review Words and compare by Awards
    features_list = ['one_star_reviews','two_star_reviews','three_star_reviews','bib_gourmand_reviews']

    fontsize = 20
    plt.rc('font', size=20)
    plt.figure(figsize=(10, 5), dpi=80)

    word_counts.sort('all_reviews', ascending=False)[features_list].head(5).plot.barh()

    plt.gca().invert_yaxis()
    plt.title('Top-5 Review words by Award', fontdict={'fontsize': fontsize})

    return plt.show()

def QMCBT_viz_2():

    # Display top Bigrams for All Review words

    # Set the plot attributes
    fontsize = 20
    plt.figure(figsize=(10, 5), dpi=80)

    # Plot
    pd.Series(nltk.bigrams(all_reviews_words)
            ).value_counts().head(5).plot.barh()
    plt.gca().invert_yaxis()

    plt.title('Top-5 Bigrams for All Review words', fontdict={'fontsize': fontsize})

    return plt.show()

def QMCBT_viz_3():

    # Display top Trigrams for All Review words

    fontsize = 20
    plt.figure(figsize=(10, 5), dpi=80)

    pd.Series(nltk.ngrams(all_reviews_words, 3)
            ).value_counts().head(5).plot.barh()
    plt.gca().invert_yaxis()
    plt.title('Top-5 Trigrams for All Review words', fontdict={'fontsize': fontsize})

    return plt.show()

def QMCBT_viz_4():

    # REVIEWS
    viz_reviews_wc_by_award = reviews_wc_by_award.sort_values(ascending=False)
    Hex_Codes_Earthy = ['#854d27', '#dd7230', '#f4c95d', '#e7e393', '#04030f']

    #create a bar plot
    plt.subplot(1,2,1)
    viz_reviews_wc_by_award.plot(kind='bar', title='Word Count of Reviews\n by Award', ylabel='',
            xlabel='',fontsize =20, color=Hex_Codes_Earthy)
    plt.xticks(rotation=45, ha='right')

    # FACILITIES
    viz_facilities_wc_by_award = facilities_wc_by_award.sort_values(ascending=False)
    Hex_Codes_Earthy = ['#854d27', '#dd7230', '#f4c95d', '#e7e393', '#04030f']

    #create a bar plot
    plt.subplot(1,2,2)
    viz_facilities_wc_by_award.plot(kind='bar', title='Word Count of Facilities\n by Award', ylabel='',
            xlabel='',fontsize =20, color=Hex_Codes_Earthy)
    plt.xticks(rotation=45, ha='right')

    return plt.show()

def stat_levene():
    # Levene
    from scipy import stats

    t_stat, p_val = stats.levene(reviews_wc_by_award, facilities_wc_by_award)

    # Set Alpha α
    α = Alpha = alpha = 0.05

    if p_val < α:
        print('equal_var = False (we cannot assume equal variance)')
        
    else:
        print('equal_var = True (we will assume equal variance)')
        
    print('_______________________________________________________________')  
    print(f't-stat: {t_stat}')
    print(f'p-value: {p_val}')

def stat_pearson():

    # Pearson's-R

    alpha = 0.05
    r, p_val = stats.pearsonr(reviews_wc_by_award, facilities_wc_by_award)
        
    if p_val < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    r= r.round(4)
    p_val = p_val.round(4)
    print('_____________________')  
    print(f'correlation {r}')
    print(f'p-value {p_val}')

###############################
##### Universal Variables #####
###############################

## ----------------------- ##
## CREATE review variables ##
## ----------------------- ##


def var_reviews(train):
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
    return all_reviews, one_star_reviews, two_star_reviews, three_star_reviews, bib_gourmand_reviews


def var_review_words():
    # Break them all into word lists with split
    all_reviews_words = all_reviews.split()
    one_star_reviews_words = one_star_reviews.split()
    two_star_reviews_words = two_star_reviews.split()
    three_star_reviews_words = three_star_reviews.split()
    bib_gourmand_reviews_words = bib_gourmand_reviews.split()
    return all_reviews_words, one_star_reviews_words, two_star_reviews_words, three_star_reviews_words, bib_gourmand_reviews_words


def var_review_freq():
    # Assign word counts to Frequency Variables
    freq_one_star_reviews = pd.Series(one_star_reviews_words).value_counts()
    freq_two_star_reviews = pd.Series(two_star_reviews_words).value_counts()
    freq_three_star_reviews = pd.Series(
        three_star_reviews_words).value_counts()
    freq_bib_gourmand_reviews = pd.Series(
        bib_gourmand_reviews_words).value_counts()
    freq_all_reviews = pd.Series(all_reviews_words).value_counts()
    return freq_one_star_reviews, freq_two_star_reviews, freq_three_star_reviews, freq_bib_gourmand_reviews, freq_all_reviews

## --------------------------- ##
## CREATE facilities variables ##
## --------------------------- ##


def var_facilities(f_train):
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
    return all_facilities, one_star_facilities, two_star_facilities, three_star_facilities, bib_gourmand_facilities


def var_facilities_words():
    # Break them all into word lists with split
    all_facilities_words = all_facilities.split()
    one_star_facilities_words = one_star_facilities.split()
    two_star_facilities_words = two_star_facilities.split()
    three_star_facilities_words = three_star_facilities.split()
    bib_gourmand_facilities_words = bib_gourmand_facilities.split()
    return all_facilities_words, one_star_facilities_words, two_star_facilities_words, three_star_facilities_words, bib_gourmand_facilities_words


def var_facilities_freq():
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
    return freq_one_star_facilities, freq_two_star_facilities, freq_three_star_facilities, freq_bib_gourmand_facilities, freq_all_facilities

## -------------------------- ##
## Create Frequency DataFrame ##
## -------------------------- ##


def word_counts():
    word_counts = pd.concat([freq_all_facilities,
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

    word_counts.columns = ['all_facilities',
                           'one_star_facilities',
                           'two_star_facilities',
                           'three_star_facilities',
                           'bib_gourmand_facilities',
                           'all_reviews',
                           'one_star_reviews',
                           'two_star_reviews',
                           'three_star_reviews',
                           'bib_gourmand_reviews']

    return word_counts

# One Function to wrangle them all
def universal_variables(train, f_train):
    """
    This Function is used to call all variables
    """
    all_reviews, one_star_reviews, two_star_reviews, three_star_reviews, bib_gourmand_reviews = var_reviews(train)
    all_reviews_words, one_star_reviews_words, two_star_reviews_words, three_star_reviews_words, bib_gourmand_reviews_words = var_review_words()
    freq_one_star_reviews, freq_two_star_reviews, freq_three_star_reviews, freq_bib_gourmand_reviews, freq_all_reviews = var_review_freq()
    all_facilities, one_star_facilities, two_star_facilities, three_star_facilities, bib_gourmand_facilities = var_facilities(f_train)
    all_facilities_words, one_star_facilities_words, two_star_facilities_words, three_star_facilities_words, bib_gourmand_facilities_words = var_facilities_words()
    freq_one_star_facilities, freq_two_star_facilities, freq_three_star_facilities, freq_bib_gourmand_facilities, freq_all_facilities = var_facilities_freq()
    word_counts = word_counts()