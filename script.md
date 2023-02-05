
Hello my name is Yuvia Cardenas your Maitre D, allow me to introduce Team Gourmand
Justin Evans your Sous Chef
Woody Sims your Pastry Chef
& Cristina Lucin your Chef de Cuisine

* Menu - Intro Yuvia
* Hors d'oeuvre - Executive Summary Yuvia
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

For the Hors d'oeuvre, here is our Executive Summary:

Some key part of Acquisition & Preparation:
- Dataset of all Michelin Awardee restaurants worldwide was acquired January 17, 2023 from Kaggle (Updated quarterly)
-  We utilized the Michelin Guide URL for each restaurant and Beautiful Soup to scrape the review text, enhancing the original data set.

Some highligts from Exploration & Modeling:
- "Bib Gourmand" was our baseline (50.3% of dataset)
- 3 Michelin star restaurants had reviews with the most words, but 3 star restaurants had the highest sentiment scores

Some key takeaways from our project:
- Restaurants with higher Michelin award levels have, on average, longer reviews
- An improvement in dining experience, seems to be the biggest driver towards a three-star restaurant review

Now to our Sous Chef Justin, for a little more on how we sourced our data


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
    * We explored several methods of NLP modeling. We elected to utilize as much useful text as possible
    removing stop words and ngrams common across multiple datasets 
* SPLIT the dataset into train, validate and test, stratifying on target of `award`
* We scaled all numeric columns for modeling

Now here's Cristina to present an exploratory Salad course:

Salade

#1 Our first question begining exploration was: What is the distribution of our target variable (award type)?

Our initial hypothesis was that in comparison all awards would have an evenly distributed slope from bib gourmand being the most while gradually declining to 3 stars. 
What we discovered was, as you can see here, the steep decline from one star to two star restaurants, and similarly, to three star Michelin restaurants.

#2 Our next question was: What countries have the most Michelin restaurants?

As you can see, France has the most Michelin restaurants in the world, likely due to it being Michelin's country of origon. A surprising discovery was that Japan had the second most Michelin restaurants, and the highest concentration of 3 star awarded restaurants (9 located in Tokoyo alone).

#3 What is the average wordcount of restaurant reviews, by award type?

Our hypothesis was confirmed, that three star reviews had a higher wordcount. As you can see there is a small difference in word count between one and two star restaurant reviews, An ANOVA test was conducted to determine if the difference in the wordcounts was significant. The statistical test confirmed that there is significance in difference for the wordcount among the awards categories.

#4 Do three star Michelin restaurants have the highest sentiment score?
We expected 3 star Michelin restaurants to have the highest sentiment score. What we found was that in fact 2 star reviews had the highest sentiment score. This finding was surprising, and leads to further questions regarding sentiment analysis and its application to culinary language.


#5 What are the most frequent words used in Michelin Restaurant reviews?
- “Modern”, “Room”, "Kitchen", "One" and "local" are the most common words
- As you can see in the charts on the right, the common bigrams and trigrams reveal some interesting combinations found in the data

#6 I'd like you to take a look at these two word clouds. On the left, is a wordcloud generated from all Bib Gourmand Reviews. On the right is a wordcloud generated from three star reviews. This graphic is a great representation of what this project has gleaned from the data--that what makes a three-star Michelin restaurant unique is the focus on service.
- Take a look on the right, you can see "Service", "Experience", "Superb", "Always"
- Take a look on the left, you can see "classic, traditional, beef, pork, chicken", a focus on ingredients, on the food
- In general, what we found is that as a restaurant's Michelin ratings increased, the words used that related to service or unique experiences increased.

Now, here's Woody to talk about today's Entree--Modeling!

Entree


For modeling first selected a baseline by which to evaluate our model, a recipe to compare our final creation to. An assumption of Bib Gourmand was chosen, as it represents roughly half of the total restaurants in our dataset. This consequentially was our accuracy score, the metric we selected for this project

The ingredients we selected from the data per our findings included the top 10 countries that showed up in the training data. We also used the word count and sentiment scores





To evaluate the body of the reviews we used a technique called TFIDF, which is a combination of two metrics:
    - Term Frequency (how often a term appears in a given document)
    and
    - Inverse Document Frequency (how often the term appears in all documents)

We Developed four different models

To further improve our models, we used a technique called Grid Search with cross validation, running different iterations of models over multiple permutations of the data to find the combination of parameters that yields the best predictions, ensuring our model was seasoned to perfection

The final dish: On unseen data, our best-performing model, Logistic Regression, was able to predict the Michelin star award with 89% accuracy, nearly 29 percent more accurate than the baseline.

 - Now to Cristina, who has this course's Data Desserts to share.

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