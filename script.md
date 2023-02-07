
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

INTRO SLIDE

Welcome to all travelers who were lead by the Michelin guide & through our shared love of food, to utilize Data Science to distill the essence of fine dining perfection. At present, a star award from the Michelin Guide is widely accepted as the pre-eminent culinary achievement of restauranteurs and chefs alike. 
Internally, Michelin preserves the integrity of the reviews by keeping reviewers (commonly called "inspectors")anonymous. Externally, "Inspectors" are strictly advised not to disclose their line of work to anyone, not even their parents.
The amount of secrecy in this process, and importance of this review in the culinary world, led our Team to ask the following question:
    "What factors can be revealed by examining Michelin restaurant reviews?"

Today for your dining pleasure Team Gourmand will serve you some delectible data we hope you enjoy.

EXECUTIVE SUMMARY SLIDE

For the Hors d'oeuvre, here is our Executive Summary:

Some key part of Acquisition & Preparation:
- Dataset of all Michelin Awardee restaurants worldwide was acquired from Kaggle (Updated quarterly)
-  We utilized the Michelin Guide URL for each restaurant and Beautiful Soup to scrape the review text, enhancing the original data set.

Some highligts from Exploration & Modeling:
- "Bib Gourmand" was our baseline (50.3% of dataset)
- 3 Michelin star restaurants had reviews with the most words, but 2 star restaurants had the highest sentiment scores

Some key takeaways from our project:
- Restaurants with higher Michelin award levels have, on average, longer reviews
- An excellent and unique dining experience is a strong driver of 3 Michelin Star award

Now to our Sous Chef Justin, for a little more on this history of Michelin and how we sourced our data:


Soup du jour:

HISTORY slide

- The Michelin guide was created in 1900 by French Brothers Édouard and André Michelin. The brothers sold tires and the guide was created to increase the amount of cars on the road in France, which at the time was estimated to be fewer than 3,000 cars.

- The guide featured information valuable for motorists, including repair shops, local hotels, and dining reviews

- Over the years, the Michelin guide has undergone many changes and developments, including adding additional award categories and expanding to over 40 countries

ACQUIRE slide

Our dataset of all Michelin Awardee restaurants worldwide was acquired from Kaggle. This dataset is updated quarterly with new additions of Michelin Awardee restaurants and removal of restaurants that no longer carry the award.  From this initial dataset, we utilized the Michelin Guide URL for each restaurant and Beautiful Soup to web-scrape the review text for each restaurant, enhancing the original dataset.

* The review text for each restaurant was then appended back to the original dataframe
* Each row represents a Michelin Awardee restaurant
* Each column represents a feature of the restaurant, including feature-engineered columns
* We acquired 6780 restaurant reviews, including 6 NaN values caused by restaurants no longer active Michelin Awardees

Aperitif
PREPARE SLIDE

Our data set was prepared following standard Data Processing procedures. Some of the preparation steps we took were:

- Dropped features not useful for this project
- Cleaned column names and values utilizing REGEX and string methods
- Feature engineering including:
    * Columns with `clean` and `lemmatized` text
    * 'word_count' representing the word count of restaurant reviews
    * A feature representing the sentiment score of the review text 
    * Missing values in the price column were imputed with the mode

- We split the dataset into train, validate and test, stratifying on target of `award`
- We scaled all numeric columns and encoded price level and country into dummy variables

Now here's Cristina to present an exploratory Salad course:


Salade

TARGET VARIABLE SLIDE

#1 Our first question begining exploration was: What is the distribution of award levels in our dataset?

- Bib Gourmand, though not a star rating, is a fourth category of Michelin award that recognizes restaurants with a simpler style of cooking. Michelin describes Bib Gourmand restaurants as ones that "leave you with a sense of satisfaction, at having eaten so well at such a reasonable price."
- One Michelin Star was the second most frequent award category, followed by two and three star restaurants. Only 2 percent of restauraunts in our dataset have received the highest, and most prestiguous, three star designation.

TOP COUNTRIES SLIDES

#2 Our next question was: What countries have the most Michelin restaurants?

France, the home of the Michelin guide and the country that is credited for intially developing Fine Dining Cuisine, has the most Michelin restaurants in the world.

TOKYO SLIDE

- A surprising discovery was that Japan had the second most Michelin restaurants, and the highest concentration of 3 star awarded restaurants (9 located in this area of Tokyo)

WORDCOUNT SLIDE

#3 What is the average wordcount of restaurant reviews, by award type?

Three star restaurant reviews have the most amount of words. As you can see there is a small difference in word count between one and two star restaurant reviews, An ANOVA test was conducted to determine if the difference in the wordcounts was significant. This test confirmed that there is a significance in difference for the wordcount between the award categories.

SENTIMENT SLIDE

#4 Do three star Michelin restaurants have the highest sentiment score?
We expected 3 star Michelin restaurants to have the highest sentiment score. What we found was that 2 star Michelin restaurants, followed by one star Michelin restaurants had the highest sentiment score. This finding was surprising, our team just assumed that naturally 3 star awards would have the highest sentiment score....but lead us to our next question:

MOST COMMON WORDS SLIDE

#5 What are the most frequent words used in Michelin Restaurant reviews?
- “Modern”, “Room”, "Kitchen", "One" and "local" are the most common words
- As you can see in the charts on the right, the common bigrams and trigrams reveal some interesting combinations found in the data
- "Pork" was the most common word in Bib Gourmand reviews, and Modern was the most common in One and Two star Michelin reviews, but Service was the most common word in Three-Star Reviews. This, combined with three-star reviews not having the highest sentiment levels, pointed us to the biggest takeaway from exploring the data.

COMPARISON SLIDE

#6 I'd like you to take a look at these two word clouds. On the left, is a wordcloud generated from Two Star Michelin Restaurant Reviews. On the right is a wordcloud generated from three star reviews. This graphic is a great representation of what this project has gleaned from the data--that what makes a three-star Michelin restaurant unique is the focus on service.
- Take a look on the right, you can see "Service", "Experience", "Superb",  "Always"
- Take a look on the left, you'll see the common words in all award levels, "Modern", "flavour", "creative", culinary", a focus on ingredients, and on the food
- In general, what we found is that as a restaurant's Michelin ratings increased, the words used that related to service or unique experiences increased. And for three star restaurants, the focus on service and experience, not just the food, was distinctly different than every other category. 
- We believe that the biggest driver towards a three star michelin review is the experience of the rater, which partly comes from delicious food, but also comes from a unique, potentially one-of-a-kind dining experience that is not easily captured in these fairly short reviews. Three star reviews speak more of the experience than the food, and the data represents that. 

Now, here's Woody to talk about today's Entree--Modeling!

Entree

MODELING SLIDE

For modeling first selected a baseline by which to evaluate our model, a recipe to compare our final creation to. An assumption of Bib Gourmand was chosen, as it represents roughly half of the total restaurants in our dataset. This consequentially was our accuracy score, the metric we selected for this project

The ingredients we selected from the data per our findings included the top 10 countries that showed up in the training data. We also used the word count and sentiment scores

To evaluate the body of the reviews we used a technique called TFIDF, which is a combination of two metrics:
    - Term Frequency (how often a term appears in a given document)
    and
    - Inverse Document Frequency (how often the term appears in all documents)

We Developed four different models

To further improve our models, we used a technique called Grid Search with cross validation, running different iterations of models over multiple permutations of the data to find the combination of parameters that yields the best predictions, ensuring our model was seasoned to perfection

The final dish: On unseen data, our best-performing model, Logistic Regression, was able to predict the Michelin star award with 89% accuracy, nearly 29 percent more accurate than the baseline.

Now to Cristina, who has this course's Data Desserts to share.

Dessert
CONCLUSION SIDE

Some of our key takeaways are:

- Restaurants with higher Michelin award levels have, on average, longer reviews
- France, Japan, and Italy have the most Michelin restaurants
- Two (2) Star Michelin Restaurant reviews have the highest sentiment levels, followed by one (1) star restaurants, three (3) star restaurants, and Bib Gourmand restaurants. However, the difference in sentiment levels between the star categories was not significant.
- Utilizing the cleaned and lemmatized text of reviews, we produced a model that predicts, with 87.9% accuracy, the award category of a restaurant.
- Our results suggest that the way Michelin reviewers talk about restaurants is impactful and meaningful. The higher level a restaurant is rated, the more service focused words, groups of two and three words occur in the review

RECOMMENDATION SLIDE

Our recommendations to Restauraunteurs, Restaurant groups, and Executive Chefs seeking to achieve Michelin designation:
- To improve your chances for Michelin designation, “shoot for the stars”--utilize the uniqueness of three star restaurant reviews in our data set to develop a business plan/model
- An improvement in dining experience, seems to be the biggest driver towards a three-star restaurant review

 Next Steps
- Pruning TF/IDF to hone model performance
- Investigate deep learning to further improve model accuracy
- Exploration of restaurant cuisine type to feature engineer
- Investigation and deeper exploration of unique words and phrases
- Clustering features for modeling

QUESTIONS