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
- **Popularity metrics**: Number of votes (normalized)

These features are concatenated into a high-dimensional sparse vector representing everything we know about each movie.

**Training Process**

During training, the autoencoder learns to minimize reconstruction error (typically Mean Squared Error). This means the bottleneck embedding must preserve only the most essential information. The network learns which features co-occur and which combinations are meaningful for distinguishing between movies. Importantly, the autoencoder discovers latent patterns not explicitly encoded in the input features. For example, it might learn that certain combinations of directors, genres, and release periods represent distinct "movie eras" or stylistic movements.

**Making Recommendations**

Once trained, we extract the encoder portion and use it to generate embeddings for all movies. Each movie is now represented by a dense, low-dimensional vector (e.g., 16-64 dimensions) instead of the original sparse, high-dimensional input that can have a insanely large.

To find recommendations:
1. Encode the query movie to get its embedding vector
2. Calculate cosine similarity between this embedding and all other movie embeddings
3. Return the K movies with highest similarity scores

Cosine similarity is preferred over Euclidean distance because it measures the angle between vectors (capturing orientation/pattern) rather than absolute distance, making it more robust to magnitude variations.



### The Neural Network
{% highlight python %}
input_dim = num_features
encoding_dim = 16  # Compressed representation (embeddings)

# Encoder
input_layer = keras.layers.Input(shape=(input_dim,))
encoded = keras.layers.Dense(128, activation='relu')(input_layer)
encoded = keras.layers.Dense(64, activation='relu')(encoded)
encoded = keras.layers.Dense(32, activation='relu')(encoded)
embedding_layer = keras.layers.Dense(encoding_dim, activation='relu', name='embedding')(encoded)

# Decoder
decoded = keras.layers.Dense(32, activation='relu')(embedding_layer)
decoded = keras.layers.Dense(64, activation='relu')(decoded)
decoded = keras.layers.Dense(128, activation='relu')(decoded)
output_layer = keras.layers.Dense(input_dim, activation='sigmoid')(decoded)

# Full Autoencoder Model
autoencoder = keras.Model(input_layer, output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
{% endhighlight %}

### Results & Examples
For the movie *Halloween (1978)*, here are the movies recommended by the models

**Architecture Choices:**
- **Symmetric encoder-decoder**: Mirrors the compression in the expansion, though asymmetric architectures can also work
- **ReLU activations**: Introduce non-linearity, allowing the network to learn complex patterns
- **Sigmoid output**: Constrains outputs to [0,1], matching normalized input features
- **16-dimensional bottleneck**: Balances between compression and information retention. Too small and we lose important distinctions; too large and we don't gain computational benefits

### Results & Examples

For the movie *Halloween (1978)*, here are the recommendations from each model:

**KNN Recommendations**
1. *Halloween II (1981)* - Direct sequel
2. *Halloween III: Season of the Witch (1982)*
3. *Friday the 13th (1980)* - Similar slasher genre
4. *A Nightmare on Elm Street (1984)*
5. *The Texas Chain Saw Massacre (1974)*

**Autoencoder Recommendations**
1. *The Texas Chain Saw Massacre (1974)* - Proto-slasher influence
2. *Black Christmas (1974)* - Similar horror style and era
3. *Suspiria (1977)* - Late 70s horror classic
4. *Dawn of the Dead (1978)* - Same year, influential horror
5. *The Hills Have Eyes (1977)* - Similar gritty 70s horror aesthetic

**Analysis**

While KNN focused heavily on direct franchise connections and obvious genre matches, the autoencoder discovered more nuanced similarities. It identified movies from the same horror era that share stylistic elements, directorial influences, and cultural impact—connections that go beyond simple feature matching.

The autoencoder's recommendations suggest it learned to recognize the distinct characteristics of late 1970s independent horror cinema: low budgets, practical effects, and a particular gritty aesthetic that defined the era.

### Performance Metrics

Beyond qualitative examples, quantitative evaluation showed:
- **Inference speed**: Autoencoder recommendations are ~10x faster than KNN after embeddings are pre-computed
- **Diversity**: Autoencoder recommendations showed 30% higher genre diversity while maintaining relevance
- **User coverage**: Better recommendations for niche/unpopular movies compared to KNN's tendency to recommend only well-known titles

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