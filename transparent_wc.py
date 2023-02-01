from PIL import Image
import numpy as np
import matplotlib.colors as mcolors
from wordcloud import WordCloud


def get_threestar_wordcloud(colors=['purple', 'white']):
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

    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color=None,
        max_words=400, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        mode='RGBA',
        height=mask.shape[0])
    wc.generate(threestar_text)
    wc.to_file('images/transparent_threestar.png')


def get_twostar_wordcloud(colors=['blue', 'red']):
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

    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color=None,
        max_words=1000, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        mode='RGBA',
        height=mask.shape[0])
    wc.generate(twostar_text)
    wc.to_file('images/transparent_twostar.png')


def get_onestar_wordcloud(colors=['firebrick', 'orangered']):
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

    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color=None,
        mode='RGBA',
        max_words=250, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        height=mask.shape[0])
    wc.generate(onestar_text)
    wc.to_file('images/transparent_onestar.png')


# Bib Gourmand Word Cloud

def get_bib_wordcloud(colors=['darkred', 'orangered']):
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

    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color=None,
        max_words=500, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        mode='RGBA',
        height=mask.shape[0])
    wc.generate(bib_text)
    wc.to_file('images/transparent_bib_gourmand.png')


def get_croissant_wordcloud(colors=['peru', 'chocolate']):
    '''
    '''
    # Import TXT file of all france words
    france_text = open("./images/all_france_words.txt",
                       mode='r', encoding='utf-8').read()
    # Import .png file of croissant image, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/croissant.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors

    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color=None, mode='RGBA',
        max_words=500, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        height=mask.shape[0])
    wc.generate(france_text)
    wc.to_file('images/transparent_croissant.png')


def get_baguette_wordcloud(colors=['peru', 'chocolate']):
    '''
    '''
    # Import TXT file of all france words
    france_text = open("./images/all_france_words.txt",
                       mode='r', encoding='utf-8').read()
    # Import .png file of baguette image, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/baguette.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors

    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color=None, mode='RGBA',
        max_words=500, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        height=mask.shape[0])
    wc.generate(france_text)
    wc.to_file('images/transparent_baguette.png')


def get_shrimp_wordcloud(colors=['darkorange', 'lightsalmon']):
    '''
    '''
    # Import TXT file of all japan words
    japan_text = open("./images/all_japan_words.txt",
                      mode='r', encoding='utf-8').read()
    # Import .png file of shrimp image, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/shrimp.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors

    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color=None,
        mode='RGBA',
        max_words=500, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        height=mask.shape[0])
    wc.generate(japan_text)
    wc.to_file('images/transparent_shrimp.png')


def get_boot_wordcloud(colors=['red', 'green']):
    '''
    '''
    # Import TXT file of all italy words
    italy_text = open("./images/all_italy_words.txt",
                      mode='r', encoding='utf-8').read()
    # Import .png file of italy boot image, create a Numpy array mask from the image
    mask = np.array(Image.open("./images/italy_boot.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors

    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color=None, mode='RGBA',
        max_words=500, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        height=mask.shape[0])
    wc.generate(italy_text)
    wc.to_file('images/transparent_boot.png')


def get_wc(txt_file, png_file, colors=['blue', 'red']):
    '''
    This function utilizes a text file of all two-star review
    words and a pre-selected image to create a word cloud containing
    all two star words in an image cloud format. It takes the text
    file and image file from the /images folder.
    '''
    # Import TXT file of all two star words
    temp_text = open(f"./images/{txt_file}.txt",
                        mode='r', encoding='utf-8').read()
    # Import .png file of three star logo, create a Numpy array mask from the image
    mask = np.array(Image.open(f"./images/{png_file}.png"))
    # replace 0 with 255 inside the mask to ensure white background
    mask[mask == 0] = 255
    # Define Colors

    custom_cmap = mcolors.ListedColormap(colors)
    # Make the wordcloud, generate the image
    wc = WordCloud(
        mask=mask, background_color=None,
        max_words=1000, max_font_size=500,
        random_state=42, width=mask.shape[1],
        colormap=custom_cmap,
        mode='RGBA',
        height=mask.shape[0])
    wc.generate(temp_text)
    wc.to_file(f'images/transparent_{png_file}.png')


if __name__ == "__main__":
    get_threestar_wordcloud()
    get_twostar_wordcloud()
    get_onestar_wordcloud()
    get_bib_wordcloud()
    get_baguette_wordcloud()
    get_croissant_wordcloud()
    get_shrimp_wordcloud()
    get_boot_wordcloud()
    get_wc()


