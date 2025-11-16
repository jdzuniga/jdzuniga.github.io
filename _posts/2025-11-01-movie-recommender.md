---
layout: post
title:  "Building a Content-Based Recommender with Autoencoders"
date:   2025-11-01
tags: regular
image: /assets/article_images/movie_recommendation/background.jpg
image2: /assets/article_images/movie_recommendation.background.jpg
thumbnail: /assets/article_images/movie_recommendation/thumbnail.svg
---
Ever scrolled endlessly hoping to find something interesting to watch? Looking for good movies or shows can often be difficult. Sometimes we want to watch content similar to what we've enjoyed before. With that in mind, I wanted to leverage machine learning to build a simple recommendation system similar to the ones that are available on popular streaming platforms like Netflix, Disney Plus, etc.

### Why Not Just Use Popularity?

The simplest approach to movie recommendations is showing people what's new and popular. A basic heuristic like "recommend the top 10 movies released this month" doesn't require machine learning or user profiling—just a sorted list by release date or box office numbers. It's fast, easy to implement, and works decently well for casual browsing.

But here's the problem: everyone gets the same recommendations. If you love action movies, you're just as likely to be recommended the latest rom-com. There's no personalization, no discovery, and frankly, no intelligence behind it.

I needed something better—a system that could understand what makes movies similar and recommend based on actual content, not just popularity or recency.

### Content-Based vs. Collaborative: Choosing the Right Approach

Two main options exist for recommendation systems: Content-Based Filtering and Collaborative Filtering.

**Content-Based Filtering** recommends items based on the characteristics of the items themselves and a user's past preferences. This is the simpler approach because it uses item features (genres, keywords, descriptions, etc.) to build a profile of what the user likes based on items they've interacted with. A model can then recommend new items similar to ones they've enjoyed before.

The advantages of this method include:
- Easy to explain why something was suggested
- No need for information from other users
- Works well for new or niche items
- Privacy-friendly (no cross-user data needed)

Some disadvantages are:
- Recommendations can become repetitive
- Requires good feature engineering
- Can't discover unexpected content outside user's known preferences

**Collaborative Filtering** recommends items based on patterns in user behavior across many users. It finds content liked by users with similar tastes, allowing for serendipitous discoveries. There's no need to understand item features explicitly, which enables unexpected recommendations.

However, this method faces challenges:
- **Cold start problem**: Without prior user data, making relevant recommendations is difficult. New users typically get only popular content
- **Popularity bias**: Unpopular content tends to never be recommended
- **Privacy concerns**: Requires collecting and analyzing user behavior data

Most modern recommendation systems like Netflix, Spotify, and Amazon use hybrid approaches that combine both methods to leverage their complementary strengths.

For this project, I chose content-based filtering because it's the most accessible when no prior user data is available. This allows anyone to build their own system without user data.

### The Dataset

To achieve this, I used IMDb [data](https://developer.imdb.com/non-commercial-datasets/) available for personal and non-commercial use. From the features available, I selected:
- Title
- Type
- Genres
- Year of release
- Runtime
- Directors
- Writers
- Average rating
- Number of ratings (vote count)

I also computed a **weighted rating** to account for both quality and popularity. This prevents obscure movies with one 10-star rating from ranking higher than acclaimed films with thousands of reviews. The formula balances the movie's average rating against the overall dataset's mean rating, weighted by the number of votes.

### The Baseline: K-Nearest Neighbors
Before tyring any complex model, it's always a good practice to establish a baseline
performance by choosing a simple model. In this case, I decided to use the K-Nearest Neighbors (KNN). The idea is straightforward: represent each movie as a point in multi-dimensional space based on its features (genres, cast, runtime, etc.), then find the K closest movies when someone asks for recommendations.

This model offered several appealing advantages for building a movie recommendation system. It's a remarkably simple to implement unsupervised model, requiring no training phase since it relies on instance-based learning. It also works immediatly when you add new data without the need of
doing batch or online training.

However, these benefits came with significant drawbacks that became apparent at scale. The algorithm is computationally expensive because finding similar movies requires calculating distances to every single movie in the database, and with thousands of movies, performance deteriorates rapidly. KNN also suffers from the curse of dimensionality: when working with many features like genres, directors, writers, etc. distance metrics lose their meaning. Even when limiting the amount of features, we can easily end up with more features than sample, which is far from ideal. Also, the time complexity is O(n) for each query. This is because for each query, we need to go through all the dasaset to find the most similar movies.

Most fundamentally, KNN doesn't learn representations or discover hidden patterns; it simply measures raw distances based on whatever features you explicitly provide, limiting its ability to uncover complex relationships in the data.


### Building the Engine: Autoencoders and Embeddings
**The Architecture**

An autoencoder is a neural network trained to learn efficient data representations in an unsupervised manner. It consists of two components:

1. **Encoder**: Progressively compresses the input through increasingly smaller layers, forcing the network to learn what information is essential
2. **Decoder**: Attempts to reconstruct the original input from the compressed representation

The key insight is the bottleneck layer—the smallest layer in the middle. This forces the network to learn a compressed representation that captures the most important patterns in the data.

**Feature Engineering**

The process begins with rich metadata engineering:
- **TF-IDF vectors** from movie titles to capture thematic content
- **Type encoding** (one-hot encoding because an item can only have one type ex: movie, show, video, etc.)
- **Genre encodings** (multi-hot encoding since movies can have multiple genres)
- **Runtime**: length in minutes of each item (normalized and log-transformed)
- **Director and writer embeddings** (using frequency-based encoding or one-hot for top contributors)
- **Temporal features**: Release year (normalized and log-transformed)
- **Quality signals**: Average rating and weighted rating (normalized)

These features are concatenated into a high-dimensional sparse vector representing everything we know about each movie. To avoid multicollinearity, `numVotes` and `averageRating` were excluded as features since this information is already captured in `weightedRating`. Including correlated features would waste bottleneck capacity by encoding redundant information, reducing the model's ability to learn other meaningful patterns.

**Training Process**

During training, the autoencoder learns to minimize reconstruction error (typically Mean Squared Error). This means the bottleneck embedding must preserve only the most essential information. The network learns which features co-occur and which combinations are meaningful for distinguishing between movies. Importantly, the autoencoder discovers latent patterns not explicitly encoded in the input features. For example, it might learn that certain combinations of directors, genres, and release periods represent distinct "movie eras" or stylistic movements.

**Making Recommendations**

Once trained, we extract the encoder portion and use it to generate embeddings for all movies. Each movie is now represented by a dense, low-dimensional vector (e.g., 16-64 dimensions) instead of the original sparse, high-dimensional input that can have a insanely large.

To find recommendations:
1. Encode the query movie to get its embedding vector
2. Calculate cosine similarity between this embedding and all other movie embeddings
3. Return the K movies with highest similarity scores

Cosine similarity is preferred over Euclidean distance because it measures the angle between vectors (capturing orientation/pattern) rather than absolute distance, making it more robust to magnitude variations.



### The Autoencoder
{% highlight python %}

input_dim = X.shape[1] 
encoding_dim = 64

# Encoder
input_layer = layers.Input(shape=(input_dim,))

encoded = layers.Dense(512, activation='relu')(input_layer) 
encoded = layers.BatchNormalization()(encoded)
encoded = layers.Dropout(0.3)(encoded)

encoded = layers.Dense(256, activation='relu')(encoded)
encoded = layers.BatchNormalization()(encoded)
encoded = layers.Dropout(0.2)(encoded)

#  Embedding layer
embedding = layers.Dense(encoding_dim, activation='linear', name='embedding')(encoded)

# Decoder
decoded = layers.Dense(256, activation='relu')(embedding)
decoded = layers.BatchNormalization()(decoded)
decoded = layers.Dropout(0.2)(decoded)

decoded = layers.Dense(512, activation='relu')(decoded) 
decoded = layers.BatchNormalization()(decoded)
decoded = layers.Dropout(0.3)(decoded)

output_layer = layers.Dense(input_dim, activation='linear')(decoded)

autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
encoder = models.Model(inputs=input_layer, outputs=embedding)

{% endhighlight %}

<img src="/../assets/article_images/movie_recommendation/diagram.png" alt=""/>

**Architecture Choices:**
- **Symmetric encoder-decoder**: Slight initial expansion followed by gradual compression preserves feature hierarchies
- **ReLU + Batch Normalization**: Non-linear learning with stable training for a ~1000 dimensional inputs
- **Linear output**: Unrestricted reconstruction for mixed continuous and binary features
- **256-dimensional bottleneck**: 4x compression captures nuanced movie similarities without information loss
- **Strategic dropout (0.2-0.3)**: Regularizes the 36k sample training set against overfitting

### Results & Examples

For the movie *Toy Story (1995)*, here are the recommendations from each model:

**Autoencoder Recommendations**

1. *Monsters, Inc. (2001)* - Well received Pixar's comedy
2. *Toy Story 2 (1999)* - Direct sequel expanding the toy universe
3. *Finding Nemo (2003)* - Pixar's adventure with emotional depth
4. *Aladdin (1992)* - Disney's animated classic
5. *Cat City (1986)* - Animated comedy with adventure

**KNN Recommendations**

1. *Toy Story 2 (1999)* - Direct sequel continuing Andy's toys' story
2. *Toy Story 3 (2010)* - Trilogy conclusion with coming-of-age themes
3. *Toy Story 4 (2019)* - Latest chapter exploring purpose and belonging
4. *Finding Nemo (2003)* - Pixar's adventure with emotional depth
5. *Up (2009)* - Pixar's adventure with heartfelt storytelling

**Analysis**

**KNN** takes a very literal approach, recommending the entire Toy Story franchise first. This makes sense given KNN's distance-based methodology and since sequels share nearly identical metadata (directors, studios, genre, cast).

**Autoencoder** shows more diversity, spreading across different Pixar and Disney animated films rather than clustering on the franchise. It seems to capture broader thematic and stylistic patterns in the animation space.

Both models stay firmly within family-friendly animation, but KNN's franchise-heavy results suggest it may be overfitting to metadata features, while the autoencoder generalizes better across similar but distinct films. While the autoencoder provides more practical recommendations by balancing similarity with diversity, some users might prefer franchise recommendations.


### Final Thoughts

This project demonstrates that autoencoders can learn meaningful movie representations from content features alone, enabling personalized recommendations without user data. The embeddings capture complex relationships that simple distance metrics miss.

**Future Improvements:**

1. **Variational Autoencoders (VAE)**: A probabilistic approach that could provide better generalization and capture uncertainty in recommendations
2. **Deeper architectures**: More layers or attention mechanisms could capture even subtler patterns
3. **Transfer learning**: Pre-trained embeddings from movie reviews or subtitles could enrich the feature space
4. **Hybrid approach**: Combining content-based embeddings with collaborative filtering (if user data becomes available) for the best of both worlds

For more in-depth details or to try it yourself, the project is available on [Github](https://github.com/jdzuniga/movie-recommender-autoencoder).


<p style="text-align: center; font-size: 15px; color: grey">
Background and thumbnail images by Freepik AI.
</p>

<!--
Pro Tip on Subtitles
Use questions when you want to hook readers: "Why Not Collaborative Filtering?"
Use statements when explaining: "Building the Autoencoder"
Use action words: "Generating," "Computing," "Discovering"
Avoid generic titles like "Implementation" or "Technical Details"
 -->

 <!-- 
 Transition Sentence Formula
"Now that we understand why, let's explore how..."
"With [previous topic] in place, the next challenge was..."
"This approach works, but how well? Let's find out."
"Before diving into results, I need to explain..."
  -->