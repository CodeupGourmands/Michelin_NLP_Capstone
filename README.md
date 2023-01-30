# <i><font size="16">Codeup Gourmands Presents...</font></i>

![My Image](/images/bib_gourmand_wordcloud.png)

# <b><i><font size="22">Michelin NLP Capstone</font></i></b>

## Project Links
* Click the buttons below to see the Project Repo and Canva presentation.  

[![GitHub](https://img.shields.io/badge/Project%20GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/CodeupGourmands/Michelin_NLP_Capstone)
[![Canva](https://img.shields.io/badge/Project%20Canva-%2300C4CC.svg?style=for-the-badge&logo=Canva&logoColor=white)](https://www.canva.com/design/DAFYvwhJqEo/gAIwb8yj4v1KezlRXNb4LA/edit?utm_content=DAFYvwhJqEo&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
[![Trello](https://img.shields.io/badge/Project%20Trello-0052CC?style=for-the-badge&logo=trello&logoColor=white)](https://trello.com/b/RCNLGlKK)

## Meet Team Codeup Gourmands!
|Yuvia Cardenas|Justin Evans|Cristina Lucin|Woodrow Sims|
|:-:|:-------------------:|:--------------------------------------------------------:|:-------------------------------------:|
|![Yuvia's_PIC](https://media.licdn.com/dms/image/D4E35AQEVdVWd6q4e0Q/profile-framedphoto-shrink_200_200/0/1673045277264?e=1675101600&v=beta&t=COJMpilKwgjdlvkMt3zjjf9LZI7ECWdrYUZ_vRkSLSM)|![Justin's_PIC](https://media.licdn.com/dms/image/D5635AQEHlbOPQOxX3g/profile-framedphoto-shrink_200_200/0/1674053257186?e=1675101600&v=beta&t=krAlXEipppaBNbcLkHXmaWFOS5AFzFBuAHZ3KGVI-kE)|![Cristina's_PIC](https://media.licdn.com/dms/image/D5603AQHE8_X2lzZ0YA/profile-displayphoto-shrink_200_200/0/1665181074756?e=1680134400&v=beta&t=_wEP-h7f7oBrY6cKyRyAgWc5xGDpnXfdU_J0ktgLPuU)|![Woody's_PIC](https://media.licdn.com/dms/image/D4E35AQEMzstSHpne1w/profile-framedphoto-shrink_200_200/0/1658367242220?e=1675101600&v=beta&t=_4-2eSZyOIixU5xtf9XUoAFJjGg_hOjB4ERDaK3yfGI)
|[![Yuvia's_LinkedIn](https://img.shields.io/badge/Yuvia's%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yuvia-cardenas-083080126/)|[![Justin's_LinkedIn](https://img.shields.io/badge/Justin's%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/qmcbt)|[![Cristina's_LinkedIn](https://img.shields.io/badge/Cristina's%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/cristina-lucin/)|[![Woody's_LinkedIn](https://img.shields.io/badge/Woodrow's%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/woodrow-sims/)
|[![Yuvia's_GitHub](https://img.shields.io/badge/Yuvia's%20GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yuvia-cardenas)|[![Justin's_GitHub](https://img.shields.io/badge/Justin's%20GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/QMCBT-JustinEvans)|[![Cristina's_GitHub](https://img.shields.io/badge/Cristina's%20GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/cristinalucin)|[![Woody's_GitHub](https://img.shields.io/badge/Woodrow's%20GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Is0metry)|

# Project Inspiration:

In 1900, there were fewer than 3,000 cars on the roads of France. In order to increase demand for cars and, accordingly, car tires, car tire manufacturers and brothers Édouard and André Michelin published a guide for French motorists, the Michelin Guide. It provided information to motorists, such as maps, car mechanics listings, petrol statins, hotels and restaurants throughout France. The guide began to award stars for fine dining establishments in 1926. Initially, there was only a single star awarded. Then, in 1931, the hierarchy of zero, one, two, and three stars was introduced. In 1955, a fourth category, "Bib Gourmand", identified restaurants with quality food at a value price.

At present, a star award from the Michelin Guide is widely accepted as the pre-eminent culinary achievement of restauranteurs and chefs alike. Michelin reviewers (commonly called "inspectors") are anonymous. Many of the company's top executives have never met an inspector; inspectors themselves are advised not to disclose their line of work, even to their parents. The amount of secrecy in this process, and importance of this review in the culinary world, led us to ask the question--"What factors can be revealed by examining Michelin restaurant reviews?"Through our shared love of food, we embarked on a journey to utilize Data Science to distill the essence of fine dining perfection."


# Project Overview:
Our Capstone Team Project utilizes Web-scraping & Natural Language Processing to develop a model that predicts Michelin food star award ratings based on content from the official Michelin review.

Following the *Data Science Pipeline*
First, our team will acquire and prepare the data for exploration. Then, we will explore the data to gain insight on which features to engineer that ultimately improve our model's accuracy. After we create several types of machine learning models that can effectly predict the Michelin food star award rating we will compare each model's performance on training and validate datasets. The model that performs the best will move forward with test dataset for final results. 


# Project Goals:
* Create a model that effectively predicts Michelin food star award ratings based on content from the official Michelin review
* Provide a well-documented jupyter notebook that contains our analysis
* Produce a Final GitHub repository
* Present a Canva slide deck suitable for a general audience which summarizes our findings and documents the results with well-labeled visualizations


# Reproduction of this Data:
Can be accomplished by simply cloning our project and running the final notebook as explained in the instructions below:

**Warning** to ensure you are not banned from the host while scraping, a 2sec sleep pause per page with a backup 5sec sleep command in case of error was implemented in the acquire function. This slows down the initial scraping run of the program. After web scraping each of the 6700+ reviews, all data is saved locally to the `michelin_df.pickle` file.

<details open="">
<summary><b>Reproduction Instructions:</b></summary><br>
<p align="left">    

* Clone the Repository using this code in your terminal ```git clone git@github.com:CodeupGourmands/Michelin_NLP_Capstone.git``` then run the ```mvp_notebook.ipynb``` Jupyter Notebook.  

* You will need to ensure the below listed files, at a minimum, are included in the repo in order to be able to run.
   * `Final_Report_NLP-Project.ipynb`
   * `acquire.py`
   * `prepare.py`
   * `explore.py`
   * `model.py`
   * `evaluate.py`  
<br>
</details>

<br>

# Initial Thoughts
Our initial thoughts are that country, cuisine, and words/groups of words (bigrams and trigrams) may be very impactful features to predict our target 'award' level. Another thought was that the price level and available facilities could also help determine the target 'award' level.

# The Plan
* Acquire initial data (CSV file) via `Kaggle` download
* Acquire review data using `Beautiful Soup` via 'get_michelin_pages' function in acquire file
* Clean and Prepare the data utilizing `RegEx` and string functions
* Explore data in search of significant relationships to target (Michelin Star Ratings) 
* Conduct statistical testing as necessary
<details open="">
<summary>▪︎ Answer 6 initial exploratory questions:</summary><br>
<p align="left">
    <b>Question 1.</b> What is the distribution of our target variable (award type)?
    <br>
    <b>Question 2.</b> What countries have the most Michelin restaurants?
    <br>
    <b>Question 3.</b> What is the average wordcount of restaurant reviews, by award type?
    <br>
    <b>Question 4.</b> Do three star Michelin restaurants have the highest sentiment score?
    <br>
    <b>Question 5.</b> What are the most frequent words used in Michelin Restaurant reviews?
    <br>
  
</details>

* Develop a Model to predict Award Category of Michelin restaurants:
    * Evaluate models on train and validate data using accuracy score
    * Select the best model based on the smallest difference in the accuracy score on the train and validate sets.
    * Evaluate the best model on test data

* Draw conclusions

# Data Dictionary:

<details open="">
<summary><b>Original Features:</b></summary><br>
<p align="left">    

|Feature    |Description       |
|:----------|:-----------------|
|name|Name of the awardee restaurant|
|address|Address of the awardee restaurant|
|location|City, country, or province of the awardee restaurant|
|price|Representation of the price value from one to four (min-max) using the curency symbol of the location country|
|cuisine|Main style of cuisine served by the awardee restaurant|
|longitude|Geographical longitude of the awardee restaurant|
|latitude|Geographical latitude of the awardee restaurant|
|url|Url address to the Michelin Review of the awardee restaurant|
|facilities_and_services|Highlighted facilities and services available at the awardee restaurant|
|data|Web-scraped review for each awardee document|
</details>

<details open="">
<summary><b>Feature Engineered:</b></summary><br>
<p align="left">    

|Feature    |Description       |
|:----------|:-----------------|
|price_level|Numeric value from 1 to 4 (min-max) representing the same relative level of expense across all countries|
|city|City as captured by the first position of the location feature|
|country|Country as captured by the second position of the location feature; also captures provinces that only had one entry in the location feature|
|clean|Tokenized text in lower case, with latin symbols only from the original data column containing the scraped reviews|
|lemmatized |Data column containing the web-scraped reviews after being cleaned and lemmatzed|
|word_count|Word count of each corresponding review|
|sentiment|Compound sentiment score of each observation|
|lem_length|Length of the lemmatized text in symbols|
|original_length|Length of the original text in symbols|
|length_diff|Difference in length between the orignal_length and the length of the `clean` text|
</details>

<details open="">
<summary><b>Target Variable:</b></summary><br>
<p align="left">    

|Feature|Value|Description       |
|:------|:---:|:-----------------|
|award|['1 michelin star', '2 michelin stars', '3 michelin stars', 'bib gourmand']|This feature identifies which award was presented to the restaurant belonging to each document|
|     |1 michelin star |"High quality cooking, worth a stop!"|
|     |2 michelin stars|"Excellent cooking, worth a detour!"|
|     |3 michelin stars|"Exceptional cuisine, worth a special journey!"|
|     |bib gourmand    |"Good quality, good value cooking."| 
</details>
<br>

# Acquire
Our dataset of all Michelin Awardee restaurants worldwide was acquired January 17, 2023 from Kaggle. This dataset is updated quarterly with new additions of Michelin Awardee restaurants and removal of restaurants that no longer carry the award.  From this initial dataset, we utilized the Michelin Guide URL for each restaurant and Beautiful Soup to web-scrape the review text for each restaurant, enhancing the original dataset.
<details open="">
<summary><b>Acquisition Actions:</b></summary><br>
<p align="left">

* Web-scraped data from `guide.michelin.com` using `Beautiful Soup`
* The review text for each restaurant was then appended back to the original dataframe
* Each row represents a Michein Awardee restaurant
* Each column represents a feature of the restaurant, including feature-engineered columns
* 6780 acquired restaurant reviews (6 NaN values caused by restaurants no longer active Michelin Awardees).
</details>
<br>
  
# Prepare
Our data set was prepared following standard Data Processing procedures and the details can be explored under the prepare actions below.

<details open="">
<summary><b>Prepare Actions:</b></summary><br>
<p align="left">

* **FEATURE ENGINEER:** Used 'bag of words' to create new categorical features from polarizing words. 
    * Created columns with `clean` and `lemmatized` text
    * Created a column containing the word_count length
    * Created a column containing sentiment score of the text 
* **DROP:** Dropped phone_number and website_url columns that contained Nulls values as determined would not be used as features for the iteration of this project. Dropped six restaurants from the original Kaggle dataset that are no longer Michelin restaurants.
* **RENAME:** Converted column names to lowercase with no spaces.    
* **ENCODED:** Features 'price_category' and 'country' were encoded into dummy variables
* **IMPUTED:** There were missing values in the price column that were imputed with the mode
* **Note:** Special care was taken to ensure that there was no leakage of this data
</details>
<br>

# Split

* **SPLIT:** train, validate and test (approx. 56/24/20), stratifying on target of `award`
* **SCALED:** We scaled all numeric columns for modeling ['lem_length','original_length','clean_length','length_diff']
* **Xy SPLIT:** split each DataFrame (train, validate, test) into X (features) and y (target) 

## Exploration Summary of Findings:

* Bib gourmand is the most common award (baseline is 50.3%), 3 Michelin stars is the least common.
* France has the most Michelin awarded restaurants, followed by Japan, Italy, U.S.A and Germany)
* Restaurants awarded 3 Michelin stars had reviews with the most words, and Bib Gourmand Restaurants had the fewest word count.
* Restaurants awarded 2 Michelin stars had the highest sentiment score, and Bib Gourmand restaurants had the lowest sentiment score 
* Most frequent single words used:
    * modern
    * room
    * wine
* Most frequent bigrams:
    * tasting menu
    * la carte
    * open kitchen
* Most frequent trigrams:
    * two tasting menu
    * take pride place

* Higher-rated restaurants had more facilities than lower rated restaurants

# Modeling

**The models created**

Used following classifier models: 
- Decision Tree 
- Random Forest 
- Logistic Regression
- Gradient Boosting Classifier

The metric used to evaluate the models was the accuracy score. The ideal model's accuracy score is expected to outperfom the baseline accuracy score.

## Modeling Summary:

Modeling Results
- We ran grid search on four different models, optimizing hyperparameters
- Logistic Regression performed the best, over both train and validate
- When run on test, Logistic Regression yielded an accuracy score of 87.9%, improving baseline accuracy by 37.6%


## Conclusions
- Restaurants with higher Michelin award levels have, on average, longer reviews
- France, Japan, and Italy have the most Michelin restaurants
- Two (2) Star Michelin Restaurant reviews have the highest sentiment levels, followed by one (1) star restaurants, three (3) star restaurants, and Bib Gourmand restaurants. However, the difference in sentiment levels between the star categories was not significant.
- Utilizing the cleaned and lemmatized text of reviews, we produced a model that predicts, with 87.9% accuracy, the award category of a restaurant.
- Our results suggest that the way Michelin reviewers talk about restaurants is impactful and meaningful, and further exploration could yield valuable results

 Recommendations
- To imrpove your chances for Michelin designation, “shoot for the stars”
- The higher level a restaurant is rated, the more service focused words, groups of two and three words occur in the review
- An improvement in dining experience, seems to be the biggest driver towards a three-star restaurant review

 Next Steps
- Pruning TF/IDF to hone model performance
- Investigate deep learning to further improve model accuracy
- Exploration of restaurant cuisine type to feature engineer
- Investigation and deeper exploration of unique words and phrases
- Clustering features for modeling
