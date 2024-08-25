# 🌳 Random Forests (or how to make a few million dollars)

As someone who first was introduced to machine learning through LLMs,  I thought that if deep learning models could handle unstructured data, it could certainly deal with structured. Why learn any other inferior, older algorithms? But I was wrong.

Not only are these models like usually better for predicting structured data (that is, given a row of some data with certain categories, predict a value), they are usually faster, require less data to train and are more accurate. How I like to think of it is that there are usually clear relationships in tabular data that we can easily pick out and describe into binary outcomes, and those binary concepts can be easily translated into machine instructions. Instead of having a model 'guess' correlations, we can directly feed them to a model.&#x20;

This is sort of a backward way of learning about machine learning because most people's introduction in this current climate is from the advanced models, but usually simple is always better. Most significantly, the decision trees can be _**clearly interpreted.**_ Not only can we use it as a predictor, we can also answer many other equally important questions besides getting a final prediction:

* What columns in the dataset are most important for our predictions?&#x20;
* How are they related to the dependent variable(the column we want to predict?)&#x20;
* Which particular features are the most important for a _particular_ observation?&#x20;

While Jeremy Howard was working on credit scoring, he was given a dataset with 7,000 columns.  The company in question had spent _millions_ of dollars on a 2-year project with a large consulting firm to figure out the factors that really mattered in predicting the credit score. In **2 hours,** on the first day of the job, he ran the data through a random forest and identified roughly the same 30 columns as the entire consulting team did. Go figure.&#x20;

The general rule of thumb in the industry is:&#x20;

* For structured data, an ensemble of decision trees (such as Random Forest or Gradient Boosting Machines) are used;&#x20;
* Whereas for unstructured data like images, audio and language deep learning deep learning is more applicable.&#x20;

The exception to structured data may be categorical variables with a large number of varying distinct labels(such as phone numbers, or zip code/coordinates) whereby the higher complexity means that deep learning becomes better at weeding out nuances with respect to changes in those categorical variables. &#x20;

To first understand Random Forests, we look at one component of the forest, a decision tree.&#x20;

## Decision Trees&#x20;

<figure><img src="../.gitbook/assets/image (1) (1).png" alt="" width="563"><figcaption></figcaption></figure>

A decision tree is a way of splitting our data with binary outcomes. In order to determine a certain outcome(which can be discrete OR continuous), the model keeps splitting the data set into smaller and smaller groups. For continuous variables like age, a threshold value is used to split the data, while for categorical variables the 'split' checks if it is equal to the category or not.  Then, each final group can be assigned a prediction score.&#x20;

The basic steps of training a DT model is:&#x20;

1. Loop through each column(attribute) of the dataset in turn.
2. For each column, loop through each possible level of that column in turn.
3. Try splitting the data into two groups, based on whether they are greater than or less than that value (or if it is a categorical variable, based on whether they are equal to or not equal to that level of that categorical variable).
4. Find the average prediction score for each of those two groups, and see how close that is to the actual prediction of each of the items of equipment in that group. That is, treat this as a very simple "model" where our predictions are simply the average of item's group.
5. After looping through all of the columns and all the possible levels for each, pick the split point that gave the best predictions using that simple model.
6. We now have two different groups for our data, based on this selected split. Treat each of these as separate datasets, and find the best split for each by going back to step 1 for each group.
7. Continue this process recursively, until you have reached some stopping criterion for each group—for instance, stop splitting a group further when it has only 20 items in it.

### Best Binary Split

There are a lot of metrics that we could use to evaluate the performance of our model after the first initial split, depending on our desired outcome(the standard ones would be accuracy, precision, recall, F1 score etc.).  The most helpful one at this stage would be to use a purity score. In other words, **what value do we split on that would give the 2 groups for which each group is most similar to others in the same group?**&#x20;

We score the purity by calculating the variance of the 2 groups, which indicates how wide the range of the dataset is, multiplied by the group size, to 'weigh' the variance:

$$
\text{Weighted Variance} = \frac{N_{\text{left}}}{N} \cdot \text{Variance}_{\text{left}} + \frac{N_{\text{right}}}{N} \cdot \text{Variance}_{\text{right}}
$$

&#x20;Then, we ask our model to always select the value to split which gives the lowest purity score, or "lowest total range of predictions in each group".&#x20;

Suppose we are looking at the[ data of the passengers onboard the Titanic,](https://www.kaggle.com/competitions/titanic/data) showing only the outcome values for simplicity. Looking at a single category, age, we want to know what age to split on (i.e above or below this value) that gives the best 2 groups. We go through all possible passenger ages and select the one which gives the lowest weighted variance.&#x20;

```
{'Sex': (0, 0.40787530982063946),
 'Age': (6.0, 0.478316717508991),
 'LogFare': (2.4390808375825834, 0.4620823937736597),
 'Class': (2, 0.46048261885806596)}
```

Here, we do the same for all the categories present: We have a dictionary of the categories to a pair of values: First, the optimal calculated value to split on, and the purity score. Since `Sex` has the lowest score, it is the best candidate among the categories to split on.&#x20;

Surprisingly, splitting on just the best variable alone gives a decent score: a Mean Squared Error of about 21%! This method is known as a variant of the [OneR](https://link.springer.com/article/10.1023/A:1022631118932) classifier; being so effective in general and easy to calculate, it makes for a very good _baseline_ model to compare future iterations against so we have a sense of what we are doing. This is a rule of thumb that I keep seeing everywhere from product management to startups to competitions: find the fastest way to get a rough sensing on how you are initially doing, then iterate as quickly as we can from there.&#x20;

### Creating the branches of the Decision Tree&#x20;

In this case, all our decision trees end up only in binary outcomes of survived vs. didn't survive. In other scenarios, decision trees may be used to assign a cost or value to each unique group that we split into, such as a confidence score for an individual's ability to pay back a loan.&#x20;

By increasing the number of branches of the trees, we can't ever decrease the accuracy of the tree(at least on the training dataset). The only downside is we run the risk of overfitting. Remember the end goal is for the model to predict outcomes for **data it has never seen before**, and the more we try to split our training dataset into smaller and smaller buckets this ability gets weaker, because the model becomes extremely sensitive to all conditions that the training data is in, even 'noise'.&#x20;

Let's take our first split, on `gender`, and ask the question: for each of our genders, what is the next best split variable for each of them?

After removing the `Sex` variable from the subset of data that was split into males, we get the following split for `Male`:

```
{'Embarked': (0, 0.3875581870410906),
 'Age': (6.0, 0.3739828371010595), ###
 'LogFare': (2.803360380906535, 0.3804856231758151),
 'Class': (1, 0.38155442004360934)}
```

And correspondingly for `female`:

```

{'Embarked': (0, 0.4295252982857327),
 'Age': (50.0, 0.4225927658431649),
 'LogFare': (4.256321678298823, 0.41350598332911376),
 'Class': (2, 0.3335388911567601)} ###
```

The splits for the lowest score(highest purity) for males is `Age<=6`, and for females is `Class<=2`.&#x20;

We take this idea, and split our tree even further:

<figure><img src="../.gitbook/assets/image (3) (1).png" alt="" width="563"><figcaption></figcaption></figure>

Another way of determining purity is through the 'Gini' score, which can be thought of as the probability that 2 random selections from the same basket give the same result every time.

There are a lot of ways to construct a tree in order to improve the overall final metric of how well the model does on our test set, like having a minimum number of leaves, trimming the length of certain branches, or setting a minimum size of each basket; however in practice these optimizations in most cases pale in comparison to the next idea, ensembling.&#x20;

If you take the predictions of a large number of uncorrelated models, which have slightly different accuracies, _their average error will end up being zero._ This is a pretty powerful insight because even without a lot of data, we can use slightly different subsets of the data to train a lot of models.&#x20;

Our strategy then will follow exactly that.&#x20;

Each time, we take a random subset of 75% of the training data and train 100 different trees. We then take the average of their predictions. We'll end up with a model that will perform better every time in the long run vs a single tree.&#x20;

A nice side effect occurs: we don't always need a separate validation set if our dataset is small. We can see how well each tree predicts the data that wasn't used to train it, then average all the trees. This is known as _out-of-bag error,_ and can give us insight into whether or not we are overfitting our data.&#x20;

## Model Interpretation

The last most interesting concept: since we can see what happens clearly at every step of the decision making process, we can use the random forest to answer important questions:

* How confident are we in our predictions using a particular row of data?
* For predicting one particular observation/sample, what were the most important factors, and how did they influence that prediction?
* Which columns are the strongest predictors, which can we ignore?
* Which columns are effectively redundant with each other, for purposes of prediction? (i.e saleWeek and saleDayofYear)
* How do predictions vary, as we vary these columns?

To answer the first question, we can take the aggregated standard deviation of all the trees on each observation(row). This tells us whether or not the randomized trees of the forest all agree with each other or not. A high standard deviation means each tree is predicting an outcome quite different, and subsequently less confident in the final prediction. This is useful to know if we want to push our model for production.



We can also see which features contribute the most to influencing the final decision to the greatest extent, i.e. _feature importance._ Let's take a look at some data with the goal of predicting the sale price of tractors.&#x20;

<figure><img src="../.gitbook/assets/image (4) (1).png" alt=""><figcaption><p>Feature importance plot for factors affecting sale price of tractors</p></figcaption></figure>

The feature importance is calculated as follows: It loops through each tree and recursively explores each branch. At each branch, it takes note of the feature used to split, and how much the model improves as a result of using that feature. The improvement(weighted again by the number of observations) is added to the importance score for that attribute, and finally the scores are normalized so that the sum adds up to 1.&#x20;

There's an important caveat in the interpretation of feature importance: It doesn't always tell the full picture. For example, in predicting the price of the tractors in the graph above, YearMade seems to indicate that newer tractors always sold for more, which seems to make sense. However, the truth is that tractors with _air conditioning_ are the ones that are pricier, but there is no column for air conditioning in our data.&#x20;

#### Removing Low-Importance Variables <a href="#removing-low-importance-variables" id="removing-low-importance-variables"></a>

let's say that after looking at feature importance, we realize that the top 10% of columns contribute to 80% of the final decision. That means that 90% of our calculations are redundant! In the real-world setting, speed and accuracy are prevalent concerns. We can establish a margin for error, for example, we can afford to lose up to 5% accuracy in return for fewer calculations, i.e. faster and cheaper compute.&#x20;

Okay, let's say we removed the majority of features that contribute little to the outcome. One thing we _didn't_ account for is that even within the group of highly important features, there can be attributes with very high correlation: i.e for the sake of our predictions they tell the same story. We can thus recursively 'combine' attributes that are similar enough, in order to further reduce the complexity of our model. This is just another step in optimizing our model and is only necessary depending on the dataset and task at hand.&#x20;

#### Partial Dependence&#x20;

Finally, we want to ask: _For all else held equal, how does our prediction vary using only this attribute?_

At first glance, this might seem like we could just take the result of each of the levels. Going back to our tractor example, to find out how the sale price of tractors varied with `yearMade`,  we might be tempted to take the average of all the observations of each year. The problem with this approach is that other factors might slightly correlate with `yearMade,` such as how many products are sold, inflation, world events, etc. Averaging over all the observations would also capture these other intrinsic relations baked into the variable, and sometimes that's not what we want.&#x20;

Instead, we replace every single value in the `yearMade` column with one specific year, and then we predict the outcome of all rows. We then repeat that for all years. In doing so, we are in some sense creating synthetic data for observations that don't exist! This isolates the effect of only `yearMade` (even if it does so by averaging over some imagined records where we assign a `yearMade` value that might never actually exist alongside some other values).



### Going Deeper/Takeaways&#x20;

I didn't plan on diverting to learn about more classic ML models like these, and here I have only scratched the surface of model and data interpretability. There is much more included in this [fastai course,](https://github.com/fastai/fastbook/blob/master/09\_tabular.ipynb) and this notebook is based on Chapter 9 of that course. . The goal was to investigate the comparisons between different types of machine learning, and the takeaway was that simpler models are easier to use, are better than deep learning models at evaluating tabular and structured data, and often offer deeper insight into the problems with the model, data, and even the framing of the problem. In doing so, I now have a greater appreciation for model explainability, which is still a huge problem for LLMs.&#x20;

This is a completely different domain than working with LLMs though. For now, I'll be staying on track with LLM architecture and applications, and someday I hope to come back and look at this topic with greater scrutiny.&#x20;



















