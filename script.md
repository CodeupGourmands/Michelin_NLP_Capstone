
Hello my name is Yuvia Cardenas your Maitre D, allow me to introduce Team Gourmand
Justin Evans your Sous Chef
Woody Sims your Pastry Chef
& Cristina Lucin your Chef de Cuisine

* Menu - Intro Yuvia
* Hors d'oeuvre - Project Goal & Overview Yuvia
* Soup du Jour - Acquire Justin
* Aperitif - Prepare Justin
* Salad - Explore Cristina 
* Entree - Modeling Woody
* Dessert - Conclusion Recommendations Cristina

Welcome to all travelers who were lead by the Michelin guide & through our shared love of food, to utilize Data Science to distill the essence of fine dining perfection. At present, a star award from the Michelin Guide is widely accepted as the pre-eminent culinary achievement of restauranteurs and chefs alike. 
Internally, the company preserves the integrity of the reviews by keeping Michelin reviewers (commonly called "inspectors")anonymous. Externally, "Inspectors" are strictly advised not to disclose their line of work to anyone, not even their parents. 
The amount of secrecy in this process, and importance of this review in the culinary world, led our Team to ask the following question:
    "What factors can be revealed by examining Michelin restaurant reviews?"

Today for your dining pleasure Team Gourmand will serve you some delectible data we hope you enjoy.


Hors d'oeuvre,

Our Capstone Team Project utilizes Web-scraping & Natural Language Processing to meet the Project Goal of developing a model that predicts Michelin food star award ratings based on content from the official Michelin review.

Following the *Data Science Pipeline* Recipe:

First, our team will acquire and prepare the data for exploration. 
Then, we will explore the data to gain insight on which features to engineer that ultimately improve our model's accuracy. After we create several types of machine learning models that can effectly predict the Michelin food star award rating we will compare each model's performance on training and validate datasets. 
Finally, the model that performs the best will move forward with test dataset for final results.


Soup du jour,

Our dataset of all Michelin Awardee restaurants worldwide was acquired January 17, 2023 from Kaggle. This dataset is updated quarterly with new additions of Michelin Awardee restaurants and removal of restaurants that no longer carry the award.  From this initial dataset, we utilized the Michelin Guide URL for each restaurant and Beautiful Soup to web-scrape the review text for each restaurant, enhancing the original dataset.

* Web-scraped data from `guide.michelin.com` using `Beautiful Soup`
* The review text for each restaurant was then appended back to the original dataframe
* Each row represents a Michein Awardee restaurant
* Each column represents a feature of the restaurant, including feature-engineered columns
* 6780 acquired restaurant reviews (6 NaN values caused by restaurants no longer active Michelin Awardees).


Aperitif

Our data set was prepared following standard Data Processing procedures and the details can be explored under the prepare actions below.

FEATURE ENGINEER: 
    * Created columns with `clean` and `lemmatized` text
    * Created a column containing the word_count length
    * Created a column containing sentiment score of the text 
    * Dropped columns not relevant for this project
    * Cleaned column names and data utilizing REGEX and string methods
    * Features 'price_category' and 'country' were encoded into dummy variables
    * There were missing values in the price column that were imputed with the mode

* SPLIT the dataset into train, validate and test, stratifying on target of `award`
* We scaled all numeric columns for modeling

Salade

#1 Our first question begining exploration was: What is the distribution of our target variable (award type)?

Our initial hypothesis was that in comparison all awards would have an evenly distributed slope from bib gourmand being the most while gradually declining to 3 stars. 
What we discovered was that the slope seemed gradual from bib gourmand to 1 star but the had a steep drop from 1 star to 2 star then gradual decline to 3 stars.

#2 Our next question was: What countries have the most Michelin restaurants?

As you can see, France has the most Michelin restaurants, likely due to it being Michelin's country of origion. A surprising discovery was that Japan had the second most Michelin restaurants, more than Italy or other European countries. 

#3 What is the average wordcount of restaurant reviews, by award type?

Our hypothesis was confirmed, that three star reviews had a higher wordcount. As you can see there is a small difference in word count between one and two star restaurant reviews, An ANOVA statistical test was conducted to determine if the difference in the wordcounts was significant. The statistical test confirmed that there is significance in difference for the wordcount among the awards categories.

#4 Do three star Michelin restaurants have the highest sentiment score?
The concensus was that the 3 star award reviews would have the highest sentiment score. What we found was that in fact 2 star reviews had the highest sentiment score. This finding was surprising, and leads to further questions regarding sentiment analysis and its application to culinary language.


#5 What are the most frequent words used in Michelin Restaurant reviews?
- “Modern”, “Room”, "Kitchen", "One" and "local" are the most common words
- As you can see in the charts on the right, the common bigrams and trigrams reveal some interesting combinations found in the data

#6 I'd like you to take a look at these two word clouds. On the left, is a wordcloud generated from all Bib Gourmand Reviews. On the right is a wordcloud generated from three star reviews. This graphic is a great representation of what this project has gleaned from the data--that what makes a three-star Michelin restaurant unique is the focus on service.
- Take a look on the right, you can see "Service", "Experience", "Superb", "Always"
- Take a look on the left, you can see "classic, traditional, beef, pork, chicken", a focus on ingredients, on the food
- In general, what we found is that as a restaurant's Michelin ratings increased, the words used that related to service or unique experiences increased.


Entree


We developed four different models using different model types: (Decision Tree, Random Forest, Logistic Regression, Gradient Boosting Classifer)

The best application (via Hyperparameter tuning) was selected for evaluation of test data.

We then created a baseline model utilizing the mode of 'Bib Gourmand' as the baseline (50.3%) of our target variable with the expectation that our models will outperform the baseline model.

We explored several methods of NLP modeling. We elected to utilize as much useful text as possible, removing a small number of stopwords and ngrams from the lemmatized dataset.

We utilized accuracy as the evaluation metric
Let's see how our models did.

Modeling Results
- We ran grid search on four different models, optimizing hyperparameters
- Logistic Regression performed the best, over both train and validate
- When run on test, Logistic Regression yielded an accuracy score of 87.9%, improving baseline accuracy by 37.6%

Dessert

Our key takeaway's are as follows:
Conclusions
- Restaurants with higher Michelin award levels have, on average, longer reviews
- France, Japan, and Italy have the most Michelin restaurants
- Two (2) Star Michelin Restaurant reviews have the highest sentiment levels, followed by one (1) star restaurants, three (3) star restaurants, and Bib Gourmand restaurants. However, the difference in sentiment levels between the star categories was not significant.
- Utilizing the cleaned and lemmatized text of reviews, we produced a model that predicts, with 87.9% accuracy, the award category of a restaurant.
- Our results suggest that the way Michelin reviewers talk about restaurants is impactful and meaningful, and further exploration could yield valuable results

 Recommendations
- To imrpove your chances for Michelin designation, “shoot for the stars”, utilize the uniqueness of three star restaurant reviews to develop a restaurant plan
- The higher level a restaurant is rated, the more service focused words, groups of two and three words occur in the review
- An improvement in dining experience, seems to be the biggest driver towards a three-star restaurant review

 Next Steps
- Pruning TF/IDF to hone model performance
- Investigate deep learning to further improve model accuracy
- Exploration of restaurant cuisine type to feature engineer
- Investigation and deeper exploration of unique words and phrases
- Clustering features for modeling