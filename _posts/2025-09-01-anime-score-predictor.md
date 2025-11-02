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

Anime have been getting more popular since the past few years. With the hype coming 
from upcoming shows and movies, predicting a show score is tricky. For creators and fans alike, having a way to forecast scores could provide valuable insight into trends and preferences.

<img src="/../assets/article_images/animeoracle//anime-market.webp" alt="Grand View Search, Anime Market (2025-2030), 2024."/>

In this project, I wanted to explore how machine learning could be applied to real-world data from the anime community. Predicting score might allow fans to understand what to expect from upcoming shows. This project focuses on predicting anime scores based on metadata such as genre, studio, producers, source title and more. The dataset was collected from MyAnimeList and contains over 20,000 anime entries. Each entry includes metadata such as title, genres, studio, number of members, favorites, and score. After testing a few models, i ended up using a LightGBM (Light Gradient Boosting Machine) which is efficient with many categorial features. I also evaluated its performance using standard regression metrics like RMSE, MAE and R².

After many iterations, the model achived an RMSE of 0.40 and an R² of 0.45. Some improvements are planned to improve the model's performance such as experimenting with neural networks to capture more complex patterns, using ensemble model's to improve accuracy and improve feature engineering.

<img src="/../assets/article_images/animeoracle/demo.png" alt="Live Website"/>

This project demonstrated that machine learning can effectively predict anime scores based on metadata. For more in depth details about this 
project, you can view the [Github](https://github.com/jdzuniga/animeoracle) repository. A [Live Demo](https://animeoracle.azurewebsites.net/)
is also available and the results are updated biweekly.

<!-- {% highlight python %}
x = 3
{% endhighlight %} -->


<p style="text-align: center; font-size: 15px; color: grey">
Background and thumbnail images by Freepik AI.
</p>

