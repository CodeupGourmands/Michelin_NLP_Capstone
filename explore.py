import acquire as a
import prepare as p
from typing import List, Union, Tuple
from typing import Union
import numpy as np
import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import matplotlib.colors as mcolors
import scipy.stats as stats
from scipy.stats import ttest_ind, levene, f_oneway
from IPython.display import Markdown as md

AWARD_COLORS = {"3 michelin stars" : "indianred", "2 michelin stars":"peachpuff",
                "1 michelin star":"lightsteelblue", "bib gourmand":"cornflowerblue"}


def get_ngram_frequency(ser: pd.Series, n: int = 1) -> pd.Series:
    '''
    Extracts ngram frequency from corpus
    ## Parameters
    ser: `Series` containing the lemmatized corpus
    n: n for ngram frequency (default 1)
    ## Returns
    a `Series` of the ngrams in corpus and their frequency
    '''
    words = ' '.join(ser).split()
    if n > 1:
        ngrams = nltk.ngrams(words, n)
        words = [' '.join(n) for n in ngrams]
    return pd.Series(words).value_counts()


def get_award_freq(train: pd.Series) -> None:
    '''
    Creates bar graph of the frequency of awards in the data.
    ## Parameters
    train: the training dataset
    ## Returns
    plots graph
    '''
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(figsize=(9, 6))
    cpt = sns.countplot(x='award',
                        data=train,
                        palette=AWARD_COLORS,
                        order=train['award'].value_counts().index)
    plt.title('Bib Gourmand is the Most Common Award Level in our Dataset')
    plt.xlabel("Award Level")
    plt.ylabel('Count of Restaurants')
    for tick in axes.xaxis.get_major_ticks():
        tick.label1.set_fontsize(10)
    plt.show()



def get_wordcount_bar(train: pd.DataFrame) -> None:
    '''
    Creates bar graph of the average wordcount of a 
    review based on the Michelin Star Award.
    ## Parameters
    train: the training dataset
    ## Returns
    plots graph
    '''
    # Use groupby to get an average length per language
    review_wordcount = train.groupby(
        'award').word_count.mean().sort_values(ascending=False)
    # Set style, make a chart
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(figsize=(9, 6))
    ax = sns.barplot(x=review_wordcount.values,
                     y=review_wordcount.index, palette=AWARD_COLORS,
                     order=['3 michelin stars', '2 michelin stars', '1 michelin star', 'bib gourmand'])
    ax.set_yticklabels(
        ['3 Michelin Stars', '2 Michelin Stars', '1 Michelin Star', 'Bib Gourmand'])
    plt.title('Average Wordcount of Michelin Star Level Restaurants')
    plt.xlabel("Average Word Count")
    plt.ylabel('Award Level')
    plt.show()


def top_10_country_viz(train: pd.DataFrame) -> None:
    '''
    Creates bar graph of top 10 countries with Michelin restaurants.
    ## Parameters
    train: the training dataset
    ## Returns
    plots graph
    '''
    # Use groupby to get an average length per language
    top_10_countries = train['country'].value_counts().head(9)
    # Set style, make a chart
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(figsize=(9, 6))
    ax = sns.barplot(x=top_10_countries.index,
                     y=top_10_countries.values,
                     palette=AWARD_COLORS)
    plt.title('Countries with the Most Michelin Restaurants')
    plt.xlabel("Countries")
    plt.ylabel('Number of Restaurants')
    plt.show()


def sentiment_scores_bar(train:pd.DataFrame)->None:
    '''
    Creates bar graph of sentiment score by award.
    ## Parameters
    train: the training dataset
    ## Returns
    plots graph
    '''
    dfg = train.groupby(['award'])['sentiment'].mean().sort_values(ascending=False)
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(figsize=(9, 6))
    ax = sns.barplot(x=dfg.index, 
                 y=dfg.values, palette=AWARD_COLORS,
                 order=['2 michelin stars', '1 michelin star', '3 michelin stars', 'bib gourmand'],
                 orient='v')
    plt.title("Two Star Restaurant Reviews Have the Highest Sentiment Scores")
    ax.set_xticklabels(
        ['2 Michelin Stars', '1 Michelin Star', '3 Michelin Stars', 'Bib Gourmand'])
    plt.xlabel("Award Category")
    plt.ylabel("Sentiment Score")
    plt.show()


def sentiment_country(train:pd.DataFrame)->None:
    '''
    Creates bar graph of sentiment score by country.
    ## Parameters
    train: the training dataset
    ## Returns
    plots graph
    '''
    dfg = train.groupby(['country'])[
        'sentiment'].mean().sort_values(ascending=False)
    # create a bar plot
    dfg.plot(kind='bar', title='Sentiment Score by Country', ylabel='Mean Sentiment Score',
             xlabel='', fontsize=10)
    plt.show()

# -----------------------------Stats Tests-------------------------------#


def get_anova_wordcount(train: pd.DataFrame) -> md:
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
    # TODO Woody fix this to spit out a Markdown object
    f, p = stats.f_oneway(train_bib.word_count, train_onestar.word_count,
                          train_twostar.word_count, train_threestar.word_count)
    if p < alpha:
        print('We reject the null hypothesis. There is sufficient\n'
              'evidence to conclude that the word count is significantly\n'
              'different between award categories.')
    else:
        print("We fail to reject the null hypothesis.")
        print(f'Test Statistic: {f}, P Statistic: {p}')


def get_stats_ttest(df: pd.DataFrame) -> md:
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

def get_threestar_wordcloud() -> None:
    '''
    This function utilizes a text file of all three-star review
    words and a pre-selected image to create a word cloud containing
    all three star words in an image cloud format. It takes the text
    file and image file from the /images folder.
    '''
    # Import TXT file of all three star words
    threestar_text = open(
        "./images/all_threestar_words.txt",
        mode='r', encoding='utf-8').read()
    # Import .png file of three star logo, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/three_stars.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors
    colors = ['purple', 'gold']
    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color=None,
        max_words=400, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        mode='RGBA',
        contour_color='gold', contour_width=2,
        height=mask.shape[0])
    wc.generate(threestar_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def get_twostar_wordcloud() -> None:
    '''
    This function utilizes a text file of all two-star review
    words and a pre-selected image to create a word cloud containing
    all two star words in an image cloud format. It takes the text
    file and image file from the /images folder.
    '''
    # Import TXT file of all two star words
    twostar_text = open("./images/all_twostar_words.txt",
                        mode='r', encoding='utf-8').read()
    # Import .png file of three star logo, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/two_stars.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors
    colors = ['blue', 'red']
    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color="lightyellow",
        max_words=1000, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        contour_color='red', contour_width=1,
        height=mask.shape[0])
    wc.generate(twostar_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def get_onestar_wordcloud() -> None:
    '''
    This function utilizes a text file of all one-star review
    words and a pre-selected image to create a word cloud containing
    all one star words in an image cloud format. It takes the text
    file and image file from the /images folder.
    '''
    # Import TXT file of all one star words
    onestar_text = open("./images/all_onestar_words.txt",
                        mode='r', encoding='utf-8').read()
    # Import .png file of three star logo, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/one_star_heart.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors
    colors = ['firebrick', 'orangered']
    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color="lightgray",
        max_words=250, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        contour_color='crimson', contour_width=1.5,
        height=mask.shape[0])
    wc.generate(onestar_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()


# Bib Gourmand Word Cloud

def get_bib_wordcloud() -> None:
    '''
    This function utilizes a text file of all bib gourmand review
    words and a pre-selected image to create a word cloud containing
    all bib gourmand words in an image cloud format. It takes the text
    file and image file from the /images folder.
    '''
    # Import TXT file of all bib gourmand star words
    bib_text = open("./images/all_bib_words.txt",
                    mode='r', encoding='utf-8').read()
    # Import .png file of bib gourmand image, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/bib_gourmand.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors
    colors = ['darkred', 'orangered']
    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color="white",
        max_words=500, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        contour_color='maroon', contour_width=1.5,
        height=mask.shape[0])
    wc.generate(bib_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def get_croissant_wordcloud()->None:
    '''
    This function utilizes a text file of all France review
    words and a pre-selected image to create a word cloud containing
    all France words in an image cloud format. It takes the text
    file and image file from the /images folder.
    '''
    # Import TXT file of all france words
    france_text = open("./images/all_france_words.txt",
                       mode='r', encoding='utf-8').read()
    # Import .png file of croissant image, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/croissant.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors
    colors = ['peru', 'chocolate']
    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color="white",
        max_words=500, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        contour_color='peru', contour_width=1,
        height=mask.shape[0])
    wc.generate(france_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def get_baguette_wordcloud()->None:
    '''
    This function utilizes a text file of all France review
    words and a pre-selected image to create a word cloud containing
    all France words in an image cloud format. It takes the text
    file and image file from the /images folder.
    '''
    # Import TXT file of all france words
    france_text = open("./images/all_france_words.txt",
                       mode='r', encoding='utf-8').read()
    # Import .png file of baguette image, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/baguette.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors
    colors = ['peru', 'chocolate']
    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color="#cccccc",
        max_words=500, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        contour_color='peru', contour_width=1.5,
        height=mask.shape[0])
    wc.generate(france_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def get_shrimp_wordcloud()->None:
    '''
    This function utilizes a text file of all Japan review
    words and a pre-selected image to create a word cloud containing
    all Japan words in an image cloud format. It takes the text
    file and image file from the /images folder.
    '''
    # Import TXT file of all japan words
    japan_text = open("./images/all_japan_words.txt",
                      mode='r', encoding='utf-8').read()
    # Import .png file of shrimp image, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/shrimp.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors
    colors = ['darkorange', 'lightsalmon']
    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color="#cccccc",
        max_words=500, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        contour_color='darkorange', contour_width=1.5,
        height=mask.shape[0])
    wc.generate(japan_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def get_boot_wordcloud():
    '''
    This function utilizes a text file of all italy review
    words and a pre-selected image to create a word cloud containing
    all italy words in an image cloud format. It takes the text
    file and image file from the /images folder.
    '''
    # Import TXT file of all italy words
    italy_text = open("./images/all_italy_words.txt",
                      mode='r', encoding='utf-8').read()
    # Import .png file of italy boot image, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/italy_boot.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors
    colors = ['red', 'green']
    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color="white",
        max_words=500, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        contour_color='red', contour_width=1,
        height=mask.shape[0])
    wc.generate(italy_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()

# TODO Justin move anything that we're not using in the final notebook into a separate file
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


##########################
##### Visualizations #####
##########################

def QMCBT_viz_wc() -> None:
    '''
    #### Description:
    Custom Function to display visualization of Most Common Review words
    #### Required Imports:
    import matplotlib as plt
    #### Parameters:
    None
    #### Returns:
    Plot
    '''
    # Set the plot attributes
    plt.rc('font', size=20)
    plt.figure(figsize=(10, 5), dpi=80)

    img = WordCloud(background_color='white'
                    ).generate(' '.join(all_reviews_words))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Most Common Review Words')

    plt.show()


def QMCBT_viz_1() -> None:
    '''
    #### Description:
    Custom Functoin to display visualization for Top-5 Review words
    #### Required Imports:
    import matplotlib as plt
    #### Parameters:
    None
    #### Returns:
    Plot
    '''
    features_list = ['one_star_reviews', 'two_star_reviews',
                     'three_star_reviews', 'bib_gourmand_reviews']

    # Set the plot attributes
    fontsize = 20
    plt.rc('font', size=20)
    plt.figure(figsize=(10, 5), dpi=80)

    word_counts_df.sort_values('all_reviews', ascending=False)[
        features_list].head(5).plot.barh()

    plt.gca().invert_yaxis()
    plt.ylabel('Top Words')
    plt.xlabel('Count of word Occurances')
    plt.title('Top-5 Review words', fontdict={'fontsize': fontsize})
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
 
    plt.show()


def QMCBT_viz_2():
    '''
    #### Description:
    Custom Function to Display visualization of Top-5 Review Bigrams
    #### Required Imports:
    import matplotlib as plt
    #### Parameters:
    None
    #### Returns:
    Plot
    '''
    # Set the plot attributes
    fontsize = 20
    plt.figure(figsize=(10, 5), dpi=80)

    # Plot
    pd.Series(nltk.bigrams(all_reviews_words)
              ).value_counts().head(5).plot.barh()
    plt.gca().invert_yaxis()
    plt.ylabel('Bigrams')
    plt.xlabel('Count of Bigram Occurances')
    plt.title('Top-5 Bigrams for All Review words',
              fontdict={'fontsize': fontsize})

    plt.show()


def QMCBT_viz_3() -> None:
    '''
    #### Description:
    Custom Function to Display visualization of Top-5 Review Trigrams
    #### Required Imports:
    import matplotlib as plt
    #### Parameters:
    None
    #### Returns:
    Plot
    '''
    # Set the plot attributes
    fontsize = 20
    plt.figure(figsize=(10, 5), dpi=80)

    pd.Series(nltk.ngrams(all_reviews_words, 3)
              ).value_counts().head(5).plot.barh()
    plt.gca().invert_yaxis()
    plt.ylabel('Trigrams')
    plt.xlabel('Count of Trigram Occurances')
    plt.title('Top-5 Trigrams for All Review words',
              fontdict={'fontsize': fontsize})

    plt.show()


def QMCBT_viz_4() -> None:
    '''
    #### Description:
    Custom Function to Display visualization for Word Count of Reviews by Award
    #### Required Imports:
    import matplotlib as plt
    import pandas as pd
    #### Parameters:
    None
    #### Returns:
    Plot
    '''
    # REVIEWS
    viz_reviews_wc_by_award = reviews_wc_by_award.sort_values(ascending=False)

    # create a bar plot
    plt.subplot(1, 2, 1)
    viz_reviews_wc_by_award.plot(kind='bar', title='Word Count of Reviews\n by Award', ylabel='',
            xlabel='',fontsize =20, color=['#ddeac1','#8e9189','#857f74','#494449'])
    plt.xticks(rotation=45, ha='right')

    # FACILITIES
    viz_facilities_wc_by_award = facilities_wc_by_award.sort_values(ascending=False)

    # create a bar plot
    plt.subplot(1, 2, 2)
    viz_facilities_wc_by_award.plot(kind='bar', title='Word Count of Facilities\n by Award', ylabel='',
            xlabel='',fontsize =20, color=['#857f74','#ddeac1','#8e9189','#494449'])
    plt.xticks(rotation=45, ha='right')

    plt.show()


def stat_levene():
    '''
    #### Description:
    Custom Function to run Levene test explicitely for this project
    #### Required Imports:
    from scipy import stats
    #### Parameters:
    None
    #### Returns:
    Print Statements
    '''
    from scipy import stats

    # Run the test and assign tstat & pval
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
    """
    #### Description:
    Custom Function to run Pearson's_R test explicitely for this project
    #### Required Imports:
    from scipy import stats
    #### Parameters:
    None
    #### Returns:
    Print Statements
    """
    # Set Alpha α
    alpha = 0.05

    # Run the test and assign tstat & pval
    r, p_val = stats.pearsonr(reviews_wc_by_award, facilities_wc_by_award)

    if p_val < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    r = r.round(4)
    p_val = p_val.round(4)
    print('_____________________')
    print(f'correlation {r}')
    print(f'p-value {p_val}')

