---
layout: post
title:  "Predicting Anime Score with Machine Learning"
date:   2025-09-2
# categories: ml lgboost anime
# tags: featured
tags: regular
image: /assets/article_images/animeoracle/background.jpg
image2: /assets/article_images/animeoracle/background.jpg
thumbnail: /assets/article_images/animeoracle/thumbnail.jpg
---
Anime has been growing in popularity over the past few years, with new shows and movies constantly creating waves of excitement among fans. Studios, streaming platforms, and advertisers are all eager to figure out which titles will capture audiences and generate popularity. Choosing the wrong project can mean wasted time and money, while a hit series can bring major profits through streaming subscriptions, merchandise, and global recognition.

With that in mind, I wanted to create a platform where anime fans and creators can get insights into which upcoming or currently airing shows are most likely to achieve high audience ratings. For fans, this model can act as a guide to discover promising titles or understand what trends are currently popular. For creators, it can highlight what factors tend to resonate with viewers. And for investors, predicting scores could point toward which anime might offer the best return on investment.

<img src="/../assets/article_images/animeoracle//anime-market.webp" alt="Grand View Search, Anime Market (2025-2030), 2024."/>

### Gathering the Data

To achieve this goal, it was necessary to have a source of data that would be updated daily in order to have access to the latest shows and movies. That is why I decided to use the MyAnimeList API, one of the largest anime and manga databases. It contains extensive anime and user information. Each record includes metadata such as:
- English/Original Title
- Genres
- Source
- Type
- Date of release
- Studios and producers
- Members and favorites count
- Average score (for released anime)

In order to use the data, some light data cleaning was performed, but more importantly, unsafe rated content was dismissed, anime with an extremely small number of members were not considered because they would add noise and irrelevant content. Also, our objective is to predict the anime score on a scale from 0 to 10. Thus, titles with unlabeled data (missing scores) were also removed.


### Avoiding Data Leaks
Preventing data leaks is always a big problem in machine learning. In this case, several variables that might seem useful at first glance had to be excluded.

For instance:
- The synopsis of unreleased anime is usually only published once the show is already airing. Thus, training a model using it would not help the model generalize to new titles with missing synopses.
- Variables like members or favorites tend to increase over time, so using them as features would unfairly improve predictions for shows that already have fan activity while also underestimating the scores of shows with a smaller fan base. 
- The number of episodes as well as the duration of each episode are also usually missing for unreleased titles, so they were not used.

To avoid temporal data leakage, the usual random train-test split was not used. Otherwise, the model would be using data from the future to predict data from the past. Instead, I applied backtesting by training the model on older data and testing it on more recent years. This better reflects real-world conditions where we predict the future based on past data.

For stability and data relevance, I used a 10-year moving window for training, ensuring the model always trained on recent trends while minimizing noise that might mislead the model. This approach is important because the anime industry has evolved significantly over time, and the industry from 2000 is very different from 2020 (different production styles, distribution channels, and audience preferences).

### Why Machine Learning?
Before considering using a machine learning model or even worse, the newest trending deep learning model, it's important to establish a simple baseline performance using a non-machine learning model so that we can justify the complexity of other models. In this case, the baseline model predicts an anime's score as the average score of its type (e.g., TV, movie, OVA). For example, we take the average of the years 2010-2019 and compare it to the year 2020. Testing the performance on 2024, the mean absolute error (MAE) was 0.70 and the root mean square error (RMSE) was 0.90, which is not a bad start for such a simple heuristic. This will help us evaluate whether our machine learning model actually provides a meaningful improvement.

### Exploring Models
After experimenting with algorithms like Random Forest, XGBoost, CatBoost, and LightGBM, I ended up choosing LightGBM (Light Gradient Boosting Machine) because of its better performance when evaluated using standard regression metrics such as RMSE, MAE, and R². This is because the model handles categorical features efficiently by finding the optimal split based on gradient statistics and works well on medium-sized datasets.

### Limitations & Challenges
Even though training the model on 10 years of data may seem like a lot of data would be available, this only gives the model a sample size under 10,000, which can be considered relatively small and wouldn't be ideal for deep neural networks. The small sample size is due to the fact that many unlabeled entries could not be used for supervised training, and outlier data were removed to help the model generalize better.

The second challenge was evaluating performance in deployment. The scores of currently airing anime fluctuate until the series finishes, making real-time evaluation tricky. This is because in many cases, even the last episode can greatly alter the current score. Also, the scores of newly aired anime converge over time, so the current score should be treated as an approximation. Thus, short-term results can make the model appear less accurate than it really is.

Finally, choosing the right features to include in the model requires both domain knowledge and experimentation. Tree-based models like LightGBM can perform feature selection by leveraging feature importance. That being said, even though LightGBM automatically finds patterns, feature engineering is crucial because it won't create interactions or infer domain-specific relationships on its own. The relationship between production metadata and audience reception isn't always straightforward, so there's always room to refine or add new features.


### Results
After multiple iterations and fine-tuning hyperparameters, the model achieved an RMSE of 0.40 and an R² of 0.45, indicating that it captures a meaningful portion of the variance in anime scores. Comparing these results to the heuristic model that achieved an MAE and RMSE of 0.7 and 0.9 respectively, we can justify the use of machine learning.

Having achieved satisfying results, some improvements that I want to explore in the future include:
- Experimenting with neural networks to capture more complex patterns.
- Using ensemble models to combine multiple predictors.
- Enhancing feature engineering with more contextual or text-based features.
- Leveraging the data obtained to create a recommendation system.


### Website Preview
<img src="/../assets/article_images/animeoracle/demo.png" alt="Live Website"/>


This project shows that machine learning can be used to forecast anime scores using only publicly available metadata. While the model isn’t perfect, it provides valuable insights into what factors may influence an anime’s reception. For more in depth details or if you want to try it yourself, the project is available on [Github](https://github.com/jdzuniga/animeoracle). Also, a [Live Demo](https://anime-oracle-96395747802.northamerica-northeast1.run.app/)
is also available and the results are updated biweekly.



<p style="text-align: center; font-size: 15px; color: grey">
Background and thumbnail images by Freepik AI.
</p>

