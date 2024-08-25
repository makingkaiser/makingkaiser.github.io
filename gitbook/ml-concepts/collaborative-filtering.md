# 🤝 Collaborative Filtering

<figure><img src="../.gitbook/assets/image (8) (1).png" alt="" width="375"><figcaption><p>Image from Nvidia</p></figcaption></figure>

Collaborative filtering is a technique used to predict ‘missing data’, such as in recommendation systems. If you already have an existing database of user preferences, based on what people have liked (Products, videos, movies), we can recommend an individual other stuff based on what he or she has liked/purchased in the past.&#x20;

The basic and simplified idea is to give every individual and product a number of _attributes_. Attributes can be the preferences of the person or the qualities of the product. Then, by matching these attributes, and given enough pairs of attributes, our algorithm can learn to predict correlations. These are known as **latent factors**, because the model is not technically aware of what these factors even are.&#x20;

These attributes can be represented as a range of values. Let’s say that we want to predict how likely a person is to like a certain movie, based on past movies that they have watched.&#x20;

We can then assign a number of factors, representing the attributes of a move, let's say _Dune._&#x20;

Assigning 3 factors  ranging from -1 to +1, with positive numbers indicating stronger matches and negative numbers weaker ones, categories being science-fiction, action, and old movies, then we could represent the movie Dune as:

```python
dune = np.array([0.98,0.9,-0.9])
```

Because it is very sci-fi, full of action, and pretty recent.

Now, we can do the exact same thing for every user, and assign a vector that has 3 factors representing each category. Let's pluck out one of our users:&#x20;

```python
user1 = np.array([0.9,0.8,-0.6])
```

Who seems to like science-y, action movies, albeit ones that are a bit more dated.

By performing a simple dot product of the vectors, we can get a score of how likely this user is to like the movie _Dune_!

```python
(user1*dune).sum()
//2.1420000000000003

```

So let’s say we have a grid of users and movies, and the values represent the ratings that were given by each user:

<figure><img src="https://lh7-us.googleusercontent.com/docsz/AD_4nXeqazUu8mN5a4F1qftmB-F3UzcWsL6e9b32xWelkFIyv9ZOFazhFR3P-GqHfBh9CDLD93Ri3Ka5jqqyzPZqUmBs0hYvMUAAE7rtChwLysG54aNFh9udmGkoj3QqYDjvR8Z3xJpH4BYlCwkOGIJX3r6MrtA?key=K566RxLMZBd4qkty-CORpQ" alt=""><figcaption></figcaption></figure>

The missing values here are movies the users haven’t seen before, and we want to try to recommend them movies that we think they’ll like. So, in theory, if we knew the preferences and attributes of each user and movie, we could give the best possible suggestions to the users.

In the 3-attribute example previously, we manually set the numbers to represent recency, genre and type. But there are countless attributes and preferences. While there are broad concepts such as genre and recency, there are others like action-oriented versus heavy on dialogue, or the presence of a specific actor. Then, how do we determine which attributes matter most for general user preferences?

The concept is the standard deep learning one: **Take an unknown concept or pattern, initialize each factor as a random vector that represents them, and let the model learn it.** By not making assumptions about what is important, we can let our model figure out which are the features that matter most in making the prediction.&#x20;

<figure><img src="https://lh7-us.googleusercontent.com/docsz/AD_4nXfJr_5pw9HkxMqljoEJ2v0-KRBHPxVv1S3uiJHor1Tt9L8OYq5B_y_0EWCgYMmIkMTQGie35TOKsirg2CPhCHfcoaMx1l11v-hKLIhYz2apci4Xnq6Sdb0MRI5Io7ps59ElajfIygo8MoHbQuHuTsma671t?key=K566RxLMZBd4qkty-CORpQ" alt=""><figcaption><p>Here, each user/movie has 4 latent factors.</p></figcaption></figure>

This is what embeddings are. We will attribute to each of our users and each of our movies a random vector of a certain length. (in practice, the length of the vector isn’t fixed. In industry, certain heuristics, based on the task at hand and problem, govern how many ‘attributes’ we want our model to learn).&#x20;

At the beginning, those numbers don't mean anything since we have chosen them randomly, but by the end of training, they will. By learning on existing data about the relations between users and movies, _without having any other information_, we will see that they still get some important features, and can isolate blockbusters from independent cinema, action movies from romance, and so on! And all that, based on just a table of numbers.

With these embeddings in place, we can optimize our parameters (that is, the latent factors) using stochastic gradient descent, to minimize the loss. At each step, the SGD optimizer will calculate the match between each movie and each user using the dot product, and will compare it to the actual rating that each user gave to each movie. It then takes the derivative, and steps the weights by multiplying by the learning rate. After many iterations, the loss will get better and better, improving the accuracy of representations.

Among the myriads of ways we could improve our model, we are missing one major factor: bias. Some users may have a predisposition to like movies, and generally rate all movies higher. Some movies might be so good or so terrible that even users who typically like or dislike movies with similar themes end up rating them drastically differently. The best way to solve this is to add a single trainable parameter to each movie/ user, that represents this level of `biasness` that each of our factors has. Instead of being _multiplied_ to the weights, it is _added_ as a flat number to each of the predictions. In this way, the overall behavior of each movie and user is captured more accurately.&#x20;

To illustrate, let's take a look at a model which has been trained with bias. For example, let's look at the movies with the lowest bias vector after training:&#x20;

```
['Children of the Corn: The Gathering (1996)',
 'Home Alone 3 (1997)',
 'Crow: City of Angels, The (1996)',
 'Mortal Kombat: Annihilation (1997)',
 'Cable Guy, The (1996)']
```

For each of these movies, even when a particular user aligns with all the latent factors(i.e. the movies has qualities that highly resonate with the user), they don’t seem to like it. And It is no surprise that these movies are the ones that are generally unpopular.&#x20;

In contrast, here at the movies with the highest bias:

```
['Titanic (1997)',
 "Schindler's List (1993)",
 'Shawshank Redemption, The (1994)',
 'Star Wars (1977)',
 'L.A. Confidential (1997)]

```

These movies were rated higher than usual, even by users who don't typically enjoy similar movies.

Looking even deeper, we can use a concept known as PCA to 'compress' our high dimensional vectors of the movies down into two, so that we can visualize how our model interprets the movies.

<figure><img src="https://lh7-us.googleusercontent.com/docsz/AD_4nXdSBvJWKtEoBL2_VrViEuH6iRn9KFBu3CzTxd1Cw5T7RoG0t9I8OHcdeuEnxY8oQ82FibLlUTXjpnGv2tQA7byEEfvqpBsQx1RxuFu2FZMp3VCH96wZGnGFwkPVgGkLHD5Qj6hWGRoKzEPxgdBfQVdWiGKn?key=K566RxLMZBd4qkty-CORpQ" alt=""><figcaption><p>We can see that our model seems to have derived some sort of representation that closely resembles a ‘classic’ vs ‘pop-culture’ grouping. Again, this was learned using only the ratings given by users! </p></figcaption></figure>

Collaborative filtering does have certain drawbacks, though. For example, how can we start to provide good recommendations when we have little or no users, meaning little pre-existing data? What about when we have to add a new user, or product?&#x20;

In addition, the type of user who gives ratings may differ from those who do not submit ratings. For example, anime lovers tend to rate anime movies frequently, compared to a more average watcher who has a wider range of tastes, but tends not to rate movies very frequently. That potentially means a large number of favorable ratings for anime movies and little to no negative reviews. In this case, representation bias is clear, but in other cases it might not be. These small changes can determine the entire direction or drift of your recommendation system.&#x20;

Because such a system is self reinforcing, these detrimental feedback loops can happen very fast, and quite often. What I am learning so far is that in any machine learning system, human oversight is still a much-needed component for services to work correctly.

Collaborative filtering was a good primer to understanding embeddings in the context of transformer models. The same concept of turning users and movies into vectors can be applied to words in a paragraph or sentence, and is a basis for NLP models. I will explore more in future entries.

&#x20; &#x20;
