## Personalized News Aggregator
##### (see full notebook [here](/docs/news_agg.html))  
  
  
**Project description:** In this project, I create a tool that saves me the time of searching through a sea of articles from multiple news sources to find those that interest me. I do so by building a smart news aggregator that automatically reads in the daily article feeds from my go-to news sources and selects those that I would be most likely to read.

First, information is collected on each article, such as the title and date published, and loaded into a dataframe. In addition to the supplied features, I use a fine-tuned BERT model that assigns a category to the article based on a short summary. To get personal preferences for the recommender, I created a script that prompts me with the title and description of each article so that I can label them with a `1` to indicate that I would read it or a `0` to indicate that I would not. Models are then trained to learn my preferences, effectively predicting the articles that I would read then suggesting them to me.

### 1. Data Collection

Most major publications have public RSS feeds that contain information on each article, including the title, description, link, source, and  publication date. Here, I access RSS feeds from my favorite news sources: *The New York Times*, *The Wall Stree Journal*, and *Financial Times*. I use [feedparser](https://feedparser.readthedocs.io/en/latest/)–designed specifically for RSS feeds–and [Beautiful Soup](https://beautiful-soup-4.readthedocs.io/en/latest/) to take the relevant information out of each feed's XML format. I then combine the data from each feed into one dataframe.  

Pictured below is a snippet from the RSS feed for the Technology section of *The New York Times*, and, more specifically, the first listed article: "OpenAI, Maker of ChatGPT, Is Trying to Grow Up". Note how its metadata is stored in a standardized set of tags, such as `<title>`, `<link>`, and `<description>`. This format is also standardized across publications, allowing libraries like feedparser to efficiently extract information from any RSS feed.

<img src="images/nyt_rss.png" style="display: block; margin: 0 auto;"/>

### 2. Feature Engineering

The relevant features provided in the metadata of the RSS feeds are shown below.

<img src="images/sample_df.png" style="display: block; margin: 0 auto;"/>

Firstly, the publication date can be compared against the current day's date to create a more informative `days_old` feature. Secondly, although *NYT* and *FT* provide data on the categories included in each article, such is not the case for *WSJ*, as seen in the third row's empty `categories` entry above. On top of this, the set of categories are not standardized between publications and also tend to be overly specific, thus limiting their generalizablility and predictive potential for a recommendation model.  
  
Existing LLMs can be used to enrich the dataset by classifying articles based on their associated textual data. I utilize a [fine-tuned BERT model](https://huggingface.co/fabriceyhc/bert-base-uncased-ag_news) trained on the [AG News dataset](https://huggingface.co/datasets/fancyzhx/ag_news), which takes in the concatenated `title` and `description` of an article and predicts one of four categories: `Business`, `Sci/Tech`, `Sports`, or `World`. In the sample articles above, BERT accurately assigns `Sci/Tech` to the first article about a brain study, `World` for the second article about emigration in Venezuela, and `Business` for the third article about the US electric vehicle market. This creates the new variable `predicted_category`–standardized across all publications–that can be used to make high-level distinctions between articles.
  
  
### 3. Automated Script for Daily Article Updates

Although I can successfully parse the article data and load it into a DataFrame, the RSS feeds are constantly updating. To ensure that all published articles are captured, the data collection functions need to be run daily. Therefore, I created a Bash script to automate the daily execution of `update_articles.py`, which does the following:  
  1) Loads in article data from the current day's RSS feeds  
  2) Calculates new features (`days_old` and `predicted_category`)  
  3) Updates the ongoing article database by merging in the current day's entries 

Here’s the Bash script used for automation:
```bash
#!/bin/bash

# Don't run the script if it has already run today
LAST_RUN_FILE="/tmp/last_update_articles_run"
TODAY=$(date +%Y-%m-%d)
if [ -f "$LAST_RUN_FILE" ] && [ "$(cat $LAST_RUN_FILE)" == "$TODAY" ]; then
    echo "Script has already run today. Exiting." >> /tmp/update_articles.log
    exit 0
fi

# Run the Python script and log the start/end times
echo "Script started at $(date)" >> /tmp/update_articles.log
/Users/jakepappo/micromamba/envs/sauron/bin/python3 /Users/jakepappo/Documents/Stuff/Projects/news_agg/update_articles.py >> /tmp/update_articles.log 2>&1
echo "Script finished at $(date)" >> /tmp/update_articles.log

# Update the last run date
echo "$TODAY" > "$LAST_RUN_FILE"
```

This script first checks to see if it has already been successfully run today. If not, it logs the start time, executes the Python script, logs the end time, and updates the last run date. This ensures that the data collection process is performed daily without manual intervention.

