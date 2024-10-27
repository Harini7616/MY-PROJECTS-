# Databricks notebook source
#!pip install nltk
#!pip install plotly
#!pip install matplotlib seaborn

# COMMAND ----------

import nltk
import string
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# COMMAND ----------

from pyspark.sql import SparkSession
df1 = spark.read.format("mongo") \
    .option("uri", "mongodb+srv://hvasu:admin@ait614.pju9rr8.mongodb.net/") \
    .option("database", "AIT_614") \
    .option("collection", "Final_Project") \
    .load()
display(df1)

# COMMAND ----------

from pyspark.sql.functions import lower, regexp_replace, col, length

# Remove duplicates
df_clean = df1.dropDuplicates()

# Handle missing values. Here we drop any rows where 'Review Text' or 'Rating' is null
df_clean = df_clean.dropna(subset=["Review Text", "Rating"])

# Normalize textual data: Convert review text to lowercase and remove punctuation
df_clean = df_clean.withColumn("Review Text", lower(col("Review Text")))
df_clean = df_clean.withColumn("Review Text", regexp_replace(col("Review Text"), "[^a-zA-Z\\s]", ""))

# Optionally, add additional features such as review text length
df_clean = df_clean.withColumn("Review Length", length(col("Review Text")))

# Show the processed DataFrame
display(df_clean)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize


# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def nltk_preprocess(text):
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove punctuation and convert to lower case
    tokens = [token.lower() for token in tokens if token not in string.punctuation]
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize words
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Rejoin tokens into a single string
    return ' '.join(tokens)

# Register UDF with Spark
nltk_preprocess_udf = udf(nltk_preprocess, StringType())

# Apply the preprocessing UDF to the DataFrame
df_clean = df_clean.withColumn("Processed Review Text", nltk_preprocess_udf(col("Review Text")))

# Show the processed DataFrame
display(df_clean)

# COMMAND ----------

#sentiment analysis 
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Define a UDF to compute sentiment score
def sentiment_score(text):
    if text:
        return sia.polarity_scores(text)['compound']
    else:
        return None

sentiment_score_udf = udf(sentiment_score, FloatType())

# Add sentiment score to DataFrame
df_clean = df_clean.withColumn("Sentiment Score", sentiment_score_udf(col("Processed Review Text")))

# Show the DataFrame with Sentiment Scores
display(df_clean)


# COMMAND ----------

#topic modelling 
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.clustering import LDA
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf, explode, split

# Tokenize text
df_clean = df_clean.withColumn("tokens", split(col("Processed Review Text"), " "))

# Define CountVectorizer and IDF for feature transformation
cv = CountVectorizer(inputCol="tokens", outputCol="raw_features")
idf = IDF(inputCol="raw_features", outputCol="features")

# Set up LDA model
lda = LDA(k=10, maxIter=10, featuresCol="features")

# Build pipeline
pipeline = Pipeline(stages=[cv, idf, lda])

# Fit model
model = pipeline.fit(df_clean)

# Transform data
result = model.transform(df_clean)

# Display topics (displaying words indices now, you might need to map back to words)
topics = model.stages[-1].describeTopics()
display(topics)

# Showing transformed features with topic distribution
result.select("Review Text", "topicDistribution").display(truncate=False)


# COMMAND ----------

#Categorizing the reviews 

from pyspark.sql.functions import count, when, col

# Define thresholds for positive, negative, and neutral sentiment
positive_threshold = 0.2
negative_threshold = -0.2

# Add a column indicating sentiment category
df_sentiment = df_clean.withColumn("Sentiment Category",
                                   when(col("Sentiment Score") > positive_threshold, "Positive")\
                                   .when(col("Sentiment Score") < negative_threshold, "Negative")\
                                   .otherwise("Neutral"))

# Group by sentiment category and calculate the count of products for each category
sentiment_counts = df_sentiment.groupBy("Sentiment Category")\
    .agg(count("*").alias("Total Count"))

# Calculate the total count of products
total_products = df_sentiment.count()

# Calculate the recommendation likelihood for each sentiment category
sentiment_recommendation = sentiment_counts.withColumn("Recommendation Likelihood",
                                                       col("Total Count") / total_products)

# Show the result
sentiment_recommendation.display()

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

# Assuming 'Rating' is already in the appropriate format, and 'Recommended IND' needs to be cast to integer
df_final = df_clean.withColumn("Recommended IND", col("Recommended IND").cast(IntegerType()))

# Calculate the correlation between Sentiment Score and Ratings
correlation_rating = df_final.stat.corr("Sentiment Score", "Rating")
print(f"Correlation between Sentiment Score and Rating: {correlation_rating}")

# Calculate the correlation between Sentiment Score and Recommendation Indicator
correlation_recommendation = df_final.stat.corr("Sentiment Score", "Recommended IND")
print(f"Correlation between Sentiment Score and Recommended IND: {correlation_recommendation}")


# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Convert Spark DataFrame to Pandas DataFrame for plotting
pandas_df = df_final.select("Sentiment Score", "Rating", "Recommended IND").toPandas()

# Scatter plot for Sentiment Score vs. Rating
plt.figure(figsize=(10, 6))
plt.scatter(pandas_df["Sentiment Score"], pandas_df["Rating"], alpha=0.5)
plt.title("Scatter Plot of Sentiment Score vs. Rating")
plt.xlabel("Sentiment Score")
plt.ylabel("Rating")
plt.grid(True)
plt.show()

# Scatter plot for Sentiment Score vs. Recommendation Indicator
plt.figure(figsize=(10, 6))
plt.scatter(pandas_df["Sentiment Score"], pandas_df["Recommended IND"], alpha=0.5, color='red')
plt.title("Scatter Plot of Sentiment Score vs. Recommended IND")
plt.xlabel("Sentiment Score")
plt.ylabel("Recommended IND")
plt.grid(True)
plt.show()


# COMMAND ----------

from pyspark.sql.functions import col, count

# Assuming 'Department Name' is the column with department details
# Calculate the number of reviews per department
top_departments = df_clean.groupBy("Department Name") \
                          .agg(count("Review Text").alias("Total Reviews")) \
                          .orderBy(col("Total Reviews").desc())  # Sort in descending order of reviews

# Show the top departments based on review counts
top_departments.display()



# COMMAND ----------

# Departments and there rating
from pyspark.sql.functions import col, avg

# Assuming 'Department Name' and 'Rating' are the columns of interest
department_ratings = df_clean.groupBy("Department Name") \
                              .agg(avg("Rating").alias("Average Rating")) \
                              .orderBy(col("Average Rating").desc())  # Sorting by average rating in descending order

# Show the results
department_ratings.display()


# COMMAND ----------

import seaborn as sns
# Calculate distribution of ratings for each department
ratings_distribution = df_clean.groupBy("Department Name", "Rating") \
                                .count() \
                                .groupBy("Department Name") \
                                .pivot("Rating") \
                                .sum("count")

# Fill null values with 0 for ratings that might not exist in some departments
ratings_distribution = ratings_distribution.na.fill(0)

# Convert to Pandas DataFrame
ratings_distribution_pandas = ratings_distribution.toPandas().set_index("Department Name")

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(ratings_distribution_pandas, annot=True, fmt="d", cmap="Blues")
plt.title("Heatmap of Ratings Distribution by Department")
plt.xlabel("Rating")
plt.ylabel("Department Name")
plt.show()


# COMMAND ----------

#Feedback Analysis for Trend department

# Filter for only the "Trend" department
trend_df = df_clean.filter(col("Department Name") == "Trend")
# Apply UDF to compute sentiment scores
trend_df = trend_df.withColumn("Sentiment Score", sentiment_score_udf(col("Processed Review Text")))
# Calculate average sentiment score and rating
trend_summary = trend_df.groupBy("Department Name") \
                        .agg(avg("Sentiment Score").alias("Average Sentiment"),
                             avg("Rating").alias("Average Rating"))
trend_summary.display()



# COMMAND ----------

from pyspark.sql.functions import col, explode, desc

# Filter for "Trend" department with low ratings (assuming ratings 1 and 2 are considered negative)
trend_negative_reviews = df_clean.filter((col("Department Name") == "Trend") & (col("Rating") <= 2))

# Explode words into separate rows using the correct column name 'tokens'
word_df = trend_negative_reviews.select(explode(col("tokens")).alias("word"))

# Count each word's occurrences
word_counts = word_df.groupBy("word").count()

# Sort the words by count in descending order and take the top 10
top_words = word_counts.orderBy(desc("count")).limit(10)

# Display the top 10 frequent words in negative reviews
top_words.display()



# COMMAND ----------

# target customers
from pyspark.sql.functions import col, desc
from pyspark.sql import functions as F

# Group by 'Age', count the number of entries per age group, and order by count in descending order
age_summary = df_clean.groupBy("Age") \
    .agg(F.count("*").alias("count")) \
    .orderBy(desc("count"))

# Get the top 10 most frequent age groups
top_ages = age_summary.limit(10)

# Display the result
top_ages.display ()

  

# COMMAND ----------

from pyspark.sql.functions import col, desc, broadcast
import plotly.express as px

# Getting top 10 ages as a list for filtering
top_ages_list = top_ages.select("Age").rdd.flatMap(lambda x: x).collect()

# Filter the original DataFrame to include only the top age groups
top_age_ratings = df_clean.filter(col("Age").isin(top_ages_list))

# Select only Age and Rating for the violin plot
age_rating_data = top_age_ratings.select("Age", "Rating")


# Convert Spark DataFrame to Pandas DataFrame for visualization
age_rating_pandas = age_rating_data.toPandas()

# Create the box plot
fig = px.box(age_rating_pandas, x="Age", y="Rating", color="Age",
             title="Box Plot of Ratings by Age Group",
             labels={"Rating": "Rating", "Age": "Age Group"})

# Enhance visual aesthetics
fig.update_layout(
    xaxis_title='Age Group',
    yaxis_title='Rating',
    plot_bgcolor='white'
)

# Optionally, adjust the colors for better visual distinction
fig.update_traces(marker=dict(size=4),
                  line=dict(width=2))

# Show the plot
fig.show()


# COMMAND ----------

#Top rated products based on class
from pyspark.sql.functions import col, avg, desc, count

# Group by 'Class Name', calculate average rating and total count of ratings, and filter out empty class names
top_rated_products = df_clean.filter(col("Class Name") != "") \
    .groupBy("Class Name") \
    .agg(
        avg("Rating").alias("Average_Rating"),
        count("Rating").alias("Total_Count")
    ) \
    .orderBy(desc("Average_Rating"))

# Display the result
top_rated_products.display()


# COMMAND ----------

#Top selling Items which needs improvement
from pyspark.sql.functions import col, avg, desc, count

# Group by 'Clothing ID', calculate total count and average rating
top_needing_improvement = df_clean.groupBy("Clothing ID") \
    .agg(
        count("Clothing ID").alias("Total_Count"),
        avg("Rating").alias("Average_Rating")
    ) \
    .filter((col("Total_Count") >= 100) & (col("Average_Rating") < 4)) \
    .orderBy("Average_Rating")

# Display the result
top_needing_improvement.display()


# COMMAND ----------

#Best Selling Items
# Group by 'Clothing ID', calculate total count of sales and average rating
best_selling_items = df_clean.groupBy("Clothing ID") \
    .agg(
        count("Clothing ID").alias("Total_Count"),
        avg("Rating").alias("Average_Rating")
    ) \
    .orderBy(desc("Total_Count")) \
    .limit(10)  # Get only the top 10 best selling items

# Display the result
best_selling_items.display()


# COMMAND ----------

#Negatives about products
# Fetch the top 10 negative words
top_negative_words = word_counts.orderBy(col("count").desc()).limit(10)

# Convert the DataFrame to a Pandas DataFrame for better formatting and display
top_negative_words_pandas = top_negative_words.toPandas()

# Display the DataFrame in a visually appealing format using Pandas
print("Top Negative Words from Reviews:")
print(top_negative_words_pandas)


