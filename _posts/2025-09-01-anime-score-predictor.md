---
layout: post
title:  "Anime Score Predictor"
date:   2025-09-2
# categories: ml lgboost anime
tags: featured
image: /assets/article_images/animeoracle/background.jpg
image2: /assets/article_images/animeoracle/background.jpg
thumbnail: /assets/article_images/animeoracle/thumbnail.jpg
---

### Predicting Anime Popularity with Machine Learning
Anime has been growing in popularity over the past few years, with new shows and movies constantly creating waves of excitement among fans. Studios, streaming platforms, and advertisers are all eager to figure out which titles will capture audiences and generate the most buzz. Choosing the wrong project can mean wasted time and money, while a hit series can bring major profits through streaming subscriptions, merchandise, and global recognition.

### Objective
The main goal of this project is to create a platform where anime fans, creators, and even investors can get insights into which upcoming or currently airing shows are most likely to achieve high audience ratings.


For fans, this model can act as a guide to discover promising titles or understand what trends might are currently popular. For creators, it can highlight what factors tend to resonate with viewers. And for investors, predicting scores could point toward which anime might offer the best return on investment.


<img src="/../assets/article_images/animeoracle//anime-market.webp" alt="Grand View Search, Anime Market (2025-2030), 2024."/>

### Model Selection
I wanted to explore how machine learning could be used on real-world entertainment data—specifically, how it could help forecast anime success. The goal of the model is to predict an anime’s score on a scale from 0 to 10.


Before jumping into complex models, it’s important to establish a simple baseline. For this project, the baseline model predicts an anime’s score as the average score of its type (e.g., TV, movie, OVA). This helps evaluate whether machine learning models actually provide a meaningful improvement. The baseline achieved an MAE of 0.70 and an RMSE of 0.90.

The dataset used comes from the MyAnimeList API, containing over 20,000 anime entries. Each record includes metadata such as:

- Title
- Genres
- Type
- Date of release
- Studios and producers
- Members and favorites count
- Final score (for released anime)

After experimenting with several algorithms, I chose LightGBM (Light Gradient Boosting Machine) because it handles categorical features efficiently and works well on medium-sized datasets. The model’s performance was evaluated using standard regression metrics such as RMSE, MAE, and R².


### Avoiding Data Leaks
Preventing data leaks is always a big problem in machine learning. In this case, several variables that might seem useful at first glance had to be excluded.

For instance:
- The synopsis of unreleased anime is usually only published once the show is already airing.
- Variables like members or favorites increase over time, so using them in training would unfairly improve predictions for shows that already have fan activity.

To avoid time-based leaks, I didn’t use a random train-test split. Instead, I applied backtesting by training the model on older data and testing it on more recent years. This better reflects real-world conditions where we predict the future based on past data.

For stability, I used a 10-year moving window for training, ensuring the model always trained on recent trends while minimizing noise. Data older than 10 years was excluded since the anime industry has evolved significantly over time.


### Limitations & Challenges
Even if trainning the data on 10 years may seem like a lot of data would be available, this only 
gives the model a sample size under 10,000. This is also due to the fact that many unlabeled could not be used for supervised trainning only and outiliers data were removed to help the model 
generalize better. 

Another challenge was evaluating performance in deployment. The scores of currently airing anime fluctuate until the series finishes, making real-time evaluation tricky. This means that short-term results can make the model appear less accurate than it really is.

Feature engineering was another major task. Choosing which variables to include requires both domain knowledge and experimentation. The relationship between production metadata and audience reception isn’t always straightforward, so there’s always room to refine or add new features.


### Results
After multiple iterations and fine-tuning, the model achieved an RMSE of 0.40 and an R² of 0.45, indicating that it captures a meaningful portion of the variance in anime scores.

Planned improvements include:
- Experimenting with neural networks to capture more complex patterns.
- Using ensemble models to combine multiple predictors.
- Enhancing feature engineering with more contextual or text-based features.


### Website Preview
<img src="/../assets/article_images/animeoracle/demo.png" alt="Live Website"/>


### Conclusion
This project shows that machine learning can be used to forecast anime scores using only publicly available metadata. While the model isn’t perfect, it provides valuable insights into what factors may influence an anime’s reception. For more in depth details or if you want to try it yourself, the project is available on [Github](https://github.com/jdzuniga/animeoracle). Also, a [Live Demo](https://animeoracle.azurewebsites.net/)
is also available and the results are updated biweekly.

<!-- {% highlight python %}
x = 3
{% endhighlight %} -->


<p style="text-align: center; font-size: 15px; color: grey">
Background and thumbnail images by Freepik AI.
</p>

