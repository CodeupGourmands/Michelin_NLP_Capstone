# <b><i><font size="20">Michelin_NLP_Capstone</font></i></b>

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

# Project Overview:
Our Capstone Team Project utilizes Web-scraping & Natural Language Processing to create a predictive model that determines Michelin food star award ratings based on content from the official Michelin review.

Following the *Data Science Pipeline*
First, our team will acquire and prepare the data for exploration. Then, we will explore the data to gain insight on which features to engineer that ultimately improve our model's accuracy. After we create several types of machine learning models that can effectly predict the Michelin food star award rating using the train and validate we will compare each model's performance. The model that performed the best will move forward with test data set for final results. 


# Project Goals:
* Create a model that effectively predicts Michelin food star award ratings based on content from the official Michelin review
* Provide a well-documented jupyter notebook that contains our analysis
* Produce a Final GitHub repository
* Present a Canva slide deck suitable for a general audience which summarizes our findings and documents the results with well-labeled visualizations.


# Reproduction of this Data:
Can be accomplished using a local `env.py` containing `github_username`, `github_token`, and host Repository link information for access to the GitHub project Readme file search results that you want to explore.
**Warning** to make the scraping successfull we added pauses 20 sec/per page. This slows down the first run of the program. After the scraping all data is saved locally in the `data.json` file.

<details open="">
<summary><b>Reproduction Instructions:</b></summary><br>
<p align="left">    

  * To retrieve a github personal access token:
    * 1. Go here and generate a personal access token: https://github.com/settings/tokens  
         You do _not_ need to select any scopes, i.e. leave all the checkboxes unchecked
    * 2. Save it in your env.py file under the variable ```github_token```  
         Add your github username to your env.py file under the variable ```github_username```  
         
* Clone the Repository using this code ```git clone git@github.com:Codeup-Mirzakhani-Group1-NLP-Project/Codeup-Mirzakhani-GitHub-Scrape-NLP-Project.git``` then run the ```Final_Report_NLP-Project.ipynb``` Jupyter Notebook. You will need to ensure the below listed files, at a minimum, are included in the repo in order to be able to run.
   * `Final_Report_NLP-Project.ipynb`
   * `acquire.py`
   * `prepare.py`
   * `explore_final.py`
   * `modeling.py`

* A step by step walk through of each piece of the Data Science pipeline can be found by reading and running the support files located in the individual team members folders on our ```Codeup-Mirzakhani-GitHub-Scrape-NLP-Project``` github repository found here: https://github.com/Codeup-Mirzakhani-Group1-NLP-Project/Codeup-Mirzakhani-GitHub-Scrape-NLP-Project
</details>

#  
# Initial Thoughts
Our initial thoughts were that since we centered our `GitHub` repositories around the topic of **Space**, that possibly unique scientific terms found within the readme files would be deterministic of the primary coding language used to conduct exploration and modeling of those projects. Another thought was that the readme files would be peppered with code specific terminology that would reveal the primary language used to code the projects.

# The Plan
* Acquire data from `GitHub` `Readme` files by scraping the `Github API`
* Clean and Prepare the data using `RegEx` and `Beautiful soup`.
* Explore data in search of relevant keyword grouping using bi-grams and n-grams 
* Conduct statistical testing as necessary
<details open="">
<summary>▪︎ Answer the following initial questions:</summary><br>
<p align="left">
    <b>Question 1.</b> How is the target variable represented in the sample?
    <br>
    <b>Question 2.</b> Are there any specific words or word groups that can assist with identifying the Language JavaScript or Java over the other languages?
    <br>
    <b>Question 3.</b> What are the top words used in cleaned C#?
    <br>
    <b>Question 4.</b> What are the most used words in cleaned python strings?
    <br>
    <b>Question 5.</b> Is there an association between coding language and the lemmatized mean length of the string?
    <br>
    <b>Question 6.</b> Is there a significant difference in Sentiment across all four languages?
    <br>
    <b>Question 7.</b> How different are the bi-grams among four programming languages?
</details>

* Develop a Model to predict program language of space related projects using either `Python`, `Javascript`, `Java`, or `C#` based on input from `GitHub` Project `Readme` files.
    * Evaluate models on train and validate data using accuracy score
    * Select the best model based on the smallest difference in the accuracy score on the train and validate sets.
    * Evaluate the best model on test data
* Run Custom Function on a single random Data input from `GitHub` `Readme` file to predict program language of that project.
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
|cousine|This is the main style of cousine served by the awardee restaurant (Some restaurants have cousine styles)|
|longitude|This is the geographical longitude of the awardee restaurant|
|latitude|This is the geographical latitude of the awardee restaurant|
|url|This is the url address to theMichelin Review of the awardee restaurant|
|facilities_and_services|Thes are the facilities and services provided by or available at the awardee restaurant|
|data|This is the scraped review for each awardee document|	
|`first_clean`| Text after cleaning the `html` and `markdown` code|


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
|     |1 michelin star |Entry level award|
|     |2 michelin stars|Mid level award  |
|     |3 michelin stars|Highest Award    |
|     |bib gourmand    |Special category award for premiere cousine provided at a low price point|. 
</details>

#  
# Acquire
Our data was acquired from the `[NAME](LINK)` GitHub repository which is updated regularly with the last update being SEP 2022 as of the completion of our project.  From this initial data set, we used the url's for each Michelin award to scrape the Reviews to enhance the original data set. The details can be seen under the acquisition actions below.
<details open="">
<summary><b>Acquisition Actions:</b></summary><br>
<p align="left">

* We scraped our data from `github.com` using `Beautiful Soup`.
* We grabbed the link of **space themed repos** where the main coding language was either `Python`, `C#`, `Java` or `Javasript` on the first 100 pages of `github`.
* Each row represents a `Readme` file from a different project repository.
* Each column represents a feature created to try and predict the primary coding languge used.
We acquired 432 entries.
</details>

#  
# Prepare
Our data set was prepared following standard Data Processing procedures and the details can be explored under the prepare actions below.

<details open="">
<summary><b>Prepare Actions:</b></summary><br>
<p align="left">

* **NULLS:** There were no null values all repositories contained a readme for us to reference
* **FEATURE ENGINEER:** Use exploration with bag of words to create new  categorical features from polarizing words. We created columns with `clean` text ,`lemmatized` text , and columns containing the lengths of them as well. We also created a column that we filled with the sentiment score of the text in the readme. 
* **DROP:** All Data acquired was used.
* **RENAME:** Columns for Human readability.    
* **REORDER:** Rearange order of columns for convenient manipulation.   
* **DROP 2:** Drop Location Reference Columns unsuitable for use with ML without categorical translation. 
* **ENCODED:** No encoding required.
* **MELT:** No melts needed.
</details>

#  
# Summary of Data Cleansing
* Luckily all of our data was usable so we had 0 nulls or drops.
    
* Note: Special care was taken to ensure that there was no leakage of this data. All code parts were removed


# Split

* **SPLIT:** train, validate and test (approx. 50/30/20), stratifying on target of `language`
* **SCALED:** We scaled all numeric columns. ['lem_length','original_length','clean_length','length_diff']
* **Xy SPLIT:** split each DataFrame (train, validate, test) into X (features) and y (target) 


## A Summary of the data

### There are 432 records (rows) in our data consisting of 1621 features (columns).
* There are 1616 categorical features
* There are 4 continuous features that represent measurements of value, size, time, or ratio.
* One of the columns contains our target feature 'language'

# Explore

* In the exploration part we tried to identify if there are words, bigrams or trigrams that could help our model to identify the programming language. 
* We ran statistical tests on the numerical features that we have created.
* We explored differences between cleaned and lemmatized versions of c# and python.
* We explore the association between coding language and the lemmatized mean of string lengths.

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
