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

At present, a star award from the Michelin Guide is widely accepted as the pre-eminent culinary achievement of restauranteurs and chefs alike. Michelin reviewers (commonly called "inspectors") are anonymous. Many of the company's top executives have never met an inspector; inspectors themselves are advised not to disclose their line of work, even to their parents. The amount of secrecy in this process, and importance of this review in the culinary world, led us to ask the question--"What factors can be revealed by examining Michelin restaurant reviews?" Through our shared love of food, we embarked on a journey to utilize Data Science to distill the essence of fine dining perfection.


# Project Overview:
Our Capstone Team Project utilizes Web-scraping & Natural Language Processing to develop a model that predicts Michelin food star award ratings based on content from the official Michelin review.

Following the *Data Science Pipeline*
First, our team will acquire and prepare the data for exploration. Then, we will explore the data to gain insight on which features to engineer that ultimately improve our model's accuracy. After we create several types of machine learning models that can effectly predict the Michelin food star award rating using the train and validate we will compare each model's performance. The model that performed the best will move forward with test data set for final results. 


# Project Goals:
* Create a model that effectively predicts Michelin food star award ratings based on content from the official Michelin review
* Provide a well-documented jupyter notebook that contains our analysis
* Produce a Final GitHub repository
* Present a Canva slide deck suitable for a general audience which summarizes our findings and documents the results with well-labeled visualizations.


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
* A step by step walk through of each piece of the Data Science pipeline can be found by reading and running the support files located in the individual team branches by visiting our ```Codeup-Mirzakhani-GitHub-Scrape-NLP-Project``` main github repository by clicking the "Project GitHub" button at the top of this readme file.
</details>

<br>

# Initial Thoughts
Our initial thoughts are that country, cuisine, and words/groups of words (bigrams and trigrams) may be very impactful features to predict our target 'award'. Another thought was that price level and available facilities could help determine the award level obtained.

# The Plan
* Acquire initial data (CSV file) via `Kaggle` download
* Acquire review data using `Beautifl Soup` via 'get_michelin_pages' function in acquire file
* Clean and Prepare the data utilizing `RegEx` and string functions
* Explore data in search of significant relationships to target (Michelin Star Ratings) 
* Conduct statistical testing as necessary
<details open="">
<summary>▪︎ Answer 10 initial exploratory questions:</summary><br>
<p align="left">
    <b>Question 1.</b> How is the target variable represented in the sample?
    <br>
    <b>Question 2.</b> Are there any specific words or word groups that can assist with identifying the Language JavaScript or Java over the other languages?
    <br>
    <b>Question 3.</b> What are the top words used in Michelin Reviews? By Award Category?
    <br>
    <b>Question 4.</b> What are the most used words in cleaned python strings?
    <br>
    <b>Question 5.</b> Is there an association between award level and the lemmatized mean length of the review?
    <br>
    <b>Question 6.</b> Is there a significant difference in Sentiment by Award Category?
    <br>
    <b>Question 7.</b> How different are the bi-grams among four programming languages?
</details>

* Develop a Model to predict Award Category of Michelin restaurants:
    * Evaluate models on train and validate data using accuracy score
    * Select the best model based on the smallest difference in the accuracy score on the train and validate sets.
    * Evaluate the best model on test data
* Run Custom Function on a single random Data input from Michelin Restaurant Review text to predict award level of that restaurant.
* Draw conclusions

# Data Dictionary:

<details open="">
<summary><b>Original Features:</b></summary><br>
<p align="left">    

|Feature    |Description       |
|:----------|:-----------------|
|name|The name of the awardee restaurant|
|address|The address of the awardee restaurant|
|location|The city and country or province of the awardee restaurant|
|price|This is a representation of the price value from one to four (min-max) using the curency symbol of the location country|
|cuisine|This is the main style of cousine served by the awardee restaurant|
|longitude|This is the geographical longitude of the awardee restaurant|
|latitude|This is the geographical latitude of the awardee restaurant|
|url|This is the url address to the Michelin Review of the awardee restaurant|
|facilities_and_services|Thes are the facilities and services provided by or available at the awardee restaurant|
|data|This is the scraped review for each awardee document|
</details>

<details open="">
<summary><b>Feature Engineered:</b></summary><br>
<p align="left">    

|Feature    |Description       |
|:----------|:-----------------|
|price_level|This is a numeric categorical of the price column from 1 to 4 (min-max) representing the same relative level of expense across all countries|
|city|This is the city as captured by the first position of the location feature|
|country|This is the country as captured by the second position of the location feature; also captures provinces that only had one entry in the location feature|
|clean|Tokenized text in lower case, with latin symbols only from the original data column containing the scraped reviews|
|lemmatized |This is the original data column containing the scraped reviews after being cleaned and lemmatzed|
|word_count|This feature shows the word count of the review each corresponding document|
|`sentiment`|The coumpound sentiment score of each observation|
|`lem_length`|The length of the lemmatized text in symbols|
|`original_length`|The length of the original text in symbols|
|`length_diff`|The difference in length between the orignal_length and the length of the `clean` text|
</details>

<details open="">
<summary><b>Target Variable:</b></summary><br>
<p align="left">    

|Feature|Value|Description       |
|:------|:---:|:-----------------|
|award|['1 michelin star', '2 michelin stars', '3 michelin stars', 'bib gourmand']|This feature identifies which award was presented to the restaurant belonging to each document|
|     |1 michelin star |"High quality cooking, worth a stop!"|
|     |2 michelin stars|"Excellent cooking, worth a detour!"|
|     |3 michelin stars|"Exceptional cuisine, worth a special journey!" |
|     |bib gourmand    |"Good quality, good value cooking"|. 
</details>
<br>

# Acquire
Our dataset of all Michelin Awardee restaurants worldwide was acquired January 17, 2023 from the Kaggle. This dataset is updated quarterly with addition/subtraction of Michelin Awardee restaurants.  From this initial data set, we utilized the Michelin Guide URL for each restaurant and Beautiful Soup to scrape the review text for reach restaurant, enhancing the original data set.
<details open="">
<summary><b>Acquisition Actions:</b></summary><br>
<p align="left">

* We scraped our data from `guide.michelin.com` using `Beautiful Soup`
* We grabbed the review text for each restaurant and appended the data back to the original dataframe
* Each row represents a Michein Awardee restaurant
* Each column represents a feature of the restaurant, including feature-engineered columns
We acquired 6780 restaurant reviews (6 NaN values caused by restaurants no longer active Michelin Awardees).
</details>
<br>
  
# Prepare
Our data set was prepared following standard Data Processing procedures and the details can be explored under the prepare actions below.

<details open="">
<summary><b>Prepare Actions:</b></summary><br>
<p align="left">

* **FEATURE ENGINEER:** Used exploration with bag of words to create new  categorical features from polarizing words. We created columns with `clean` text ,`lemmatized` text , and a column containing the word_count length of the reviews as well. We also created a column that we filled with the sentiment score of the text in the reviews. 
* **DROP:** We dropped two columns that contained Nulls because we determined they would not be used as features for this itteration of our project.
* **RENAME:** Columns to lowercase with no spaces.    
* **ENCODED:** For modeling, price_category and country were encoded into dummy variables
</details>
<br>
  
# Summary of Data Cleansing
[EXPLICITLY DISCUSS NULLS & IMPUTING]
* **NULLS:** There were no Nulls in our Target feature (award), we dropped phone_number and website_url features since they contained nulls and we did not need them for this iteration of the project. Six restaurants from the original Kaggle dataset are no longer Michelin restaurants, and those establishments were dropped
* **IMPUTED:** There were missing values in the price column that were imputed with the mode
* **Note:** Special care was taken to ensure that there was no leakage of this data

# Split

* **SPLIT:** train, validate and test (approx. 56/24/20), stratifying on target of `award`
* **SCALED:** We scaled all numeric columns for modeling ['lem_length','original_length','clean_length','length_diff']
* **Xy SPLIT:** split each DataFrame (train, validate, test) into X (features) and y (target) 

## A Summary of the data

### There are 6780 records (rows) in our data consisting of XXXX features (columns).
* There are XXXX categorical features
* There are XXXX continuous features that represent measurements of value, size, time, or ratio.
* One of the columns contains our target feature 'award'

# Explore

* In the exploration part we tried to identify if there are words, bigrams or trigrams that could help our model to identify the award category
* We ran statistical tests on the numerical and categorical features that we have created
* We explore the association between award category and the lemmatized mean of review length

## Exploration Summary of Findings:
* In the space thematic Javascript is the most popular language. It makes up 35% of the data sample.
* Most popular "word" in **C#** is `&#9;`.
* The word `codeblock` appears only in **Python** repositories. 
* Most used in **Python** is `python`.
* The words that identifies **Java** most are `x` and `planet`.
* Most appearing bigram in **Javascript** is "bug fixed".
* Bi-grams different a lot among the programming languages `Readme` files, but the number of most occuring bi-grams is not big enough to use them in our modeling.
* There is *no significant difference* in the length of the lemmatized text among the languages.
* There is *no significant difference* in the compound sentiment score among the languages.

# Modeling

### Features that will be selected for Modeling:
* All continious variables:
    - `sentiment`
    - `lem_length`
    - `original_length`
    - `length_diff`
* `lemmatized` text turned into the Bag of Words with `TDIFVectorizer`

### Features we didn't include to modeling
* `original` 
* `first_clean`
* `clean`

Those features were used in the exploration and do not serve for the modeling.
N-grams were not created for the modeling.

**The models we created**

We used following classifiers (classification algorithms): 
- Decision Tree, 
- Random Forest, 
- Logistic Regression,
- Gaussian NB,
- Multinational NB, 
- Gradient Boosting, and
- XGBoost. 

For most of our models we have used `GridSearchCV` algorithm that picked the best feature combinations for our training set. The parameters that we've used you can see below.

To evaluate the models we used the accuracy score. The good outcome is the one where the `accuracy score` is higher than our `baseline` - the propotion of the most popular programming language in our train data set. It is `JavaScript` and `0.35`. So our baseline has the accuracy score - 0.35

## Modeling Summary:
- The best algorithm  is `Random Forest Classifier` with following parameters `{'max_depth': 5, 'min_samples_leaf': 3}`
- It predicts the programming language with accuracy:
    - 63% on the train set
    - 48% on the validate set
    - 59% on the test set
- It makes 24% better predictions on the test set that the baseline model.


# Conclusions: 
*The goals of the project were:*
- Scrape Readme files from the GitHub repositories.
- Analyzie the data.
- Build the classification model that can predict the programming language of the repository with the accuracy score higher than 35%.

*Result:*

- Even with the small amount of unorganized data it is possible to perform an exploraion analysis and identify the words that are helping to predict the programming language. We pulled 432 GitHub reposittories with the space theme in four programming languages: Javascript, Java, Python and C#.
- We built the model that has an accuracy scoreof 59%.
- Despite the statistical tests showed that engineered features are not significant, one of our models showed their importance and adding those features to our modeling improved a bit its performance of all of models.


## **Recommendations and next steps:**
- Retrieve more data to train the model on and potentially identify better features for the model.
- Possibly perform less cleaning, leave the code in markdown.
- Collect data on more languages to give the model more utility after production.
