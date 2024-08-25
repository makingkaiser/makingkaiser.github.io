---
layout:
  title:
    visible: true
  description:
    visible: true
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
---

# 🐥 NLP for Dummies

NLP refers to the applications where a model attempts to understand text to achieve certain outcomes, such as sentiment classification, translation and question answering.

This is a practical guide with code, but the aim is not to learn the details but rather why we do the things we do.

The cool thing about a lot of problems which do not _seem_ to be NLP tasks can actually be transformed into them. For instance, In the [U.S. Patent Phrase to Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/) competition on Kaggle, we want to compare 2 different words or short phrases, and, depending on which patent class they are in, score them based on how similar they are.&#x20;

Lets look at our data from the Kaggle competition:

```python
df = pd.read_csv(path/'train.csv')
```

<figure><img src="../.gitbook/assets/image (6) (1).png" alt=""><figcaption></figcaption></figure>

Here, _anchor_ and _target_ refer to the phrases we are comparing, the _score_ is the % similarity of the phrase, and _context_ is the id for the patent class they were used in.

A score of `1` means the phrases are almost synonymous ,and `0` means they have totally different meanings. _Abatement_ and _eliminating process_ have a score of `0.5`, meaning they're somewhat similar, but not identical. We can turn this into a classification task by asking the question:

“For {Phrase 1} and {Phrase 2}, choose a level of similarity: Different; Similar; or Identical".

Thus, we might be able to transform this task by transforming our input into the model into something like "_TEXT1: abatement; TEXT2: eliminating process_". Let’s add a row to our `df` for this input.

```python
df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor

```

```python
df.input.head()

### the output ###
0    TEXT1: A47; TEXT2: abatement of pollution; ANC...
1    TEXT1: A47; TEXT2: act of abating; ANC1: abate...
2    TEXT1: A47; TEXT2: active catalyst; ANC1: abat...
3    TEXT1: A47; TEXT2: eliminating process; ANC1: ...
4    TEXT1: A47; TEXT2: forest region; ANC1: abatement
Name: input, dtype: object
```

Next, we’ll have to feed our data into the model in a way that it understands. For NLP, that means preparing our `Dataset` object.

```python
from datasets import Dataset,DatasetDict

ds = Dataset.from_pandas(df)
ds
### the output ###
Dataset({
    features: ['id', 'anchor', 'target', 'context', 'score', 'input'],
    num_rows: 36473
})
```

Our sentences of inputs have to be transformed into numbers, and currently the best way to do that is through a process called _**tokenization**_. In short, since we want to reduce the number of variables that go in to the decision making process of the model, we split our words into appropriate chunks in the most efficient way that the model has learned. The best way to do that is to segment letters or words into ‘tokens’. Different models have different ways of tokenizing vocabulary, and it depends on which model we use.

When you use ChatGPT or other LLMs, this is done automatically for you. But here, lets take a look under the hood to get a better understanding of how it works.

First, we pick the model we want to use from HuggingFace, and import our high level classes that we need. Note that our tokenizer and model are presets from a base model, 'microsoft/deberta-v3-small'. Again, the details of the code is not the focus, but rather the concept behind the processes.

```python
model_nm = 'microsoft/deberta-v3-small'
from transformers import AutoModelForSequenceClassification,AutoTokenizer
tokz = AutoTokenizer.from_pretrained(model_nm)

#we then define a function which tokenizes our inputs
def tok_func(x): return tokz(x["input"])

#and in order to run this quickly in parallel on every row in our dataset, we use `map`:
tok_ds = ds.map(tok_func, batched=True)
```

```python
tokz.tokenize("My name is Kaiser, and I like cats")
### the output ###
['▁My', '▁name', '▁is', '▁Kaiser', ',', '▁and', '▁I', '▁like', '▁cats'

```

Words that are less common will be split into smaller chunks, and starting words are demarcated by a `-`.

```python
tokz.tokenize("A platypus is an ornithorhynchus anatinus.")
### the output ###
['▁A',
 '▁platypus',
 '▁is',
 '▁an',
 '▁or',
 'ni',
 'tho',
 'rhynch',
 'us',
 '▁an',
 'at',
 'inus',
 '.']
```

In other languages or syntax, like code, depending on how the model is built, the way text is split may differ. However, trying to optimize this step doesn’t usually make a huge difference in model performance; so people usually stick to best practices.

Next, we prep our labels. Transformers from HuggingFace always assume that our labels have the column `labels`, so we just rename our column in our `tok_ds` object we created when tokenizing our data.

```python
tok_ds = tok_ds.rename_columns({'score':'labels'})
```

Now that we have our tokens and labels, let’s move onto splitting our dataset. According to a lot of people smarter than me, this is one of the most important concepts of machine learning.

To illustrate, let’s say we want to fit a model where the true relationship is a quadratic.

<figure><img src="../.gitbook/assets/image (7) (1).png" alt="" width="375"><figcaption></figcaption></figure>

In real life, however, we can only take fixed samples, which can contain some noise and inaccuracy.

<figure><img src="../.gitbook/assets/Untitled 2 (3).png" alt="" width="375"><figcaption></figcaption></figure>

Over or under-fitting means that our model makes predictions about data that is either too general, or too specific to the training examples. Suppose we have the most general prediction, a line:

<figure><img src="../.gitbook/assets/Untitled 3 (3).png" alt="" width="375"><figcaption></figcaption></figure>

The points on the line aren’t close to our data at all. We have _under-fit_ our model for our data. Usually, under-fitting is quite easy to spot and resolve.

However, let’s look at what happen if we try our absolute best to fit our curve to the model (handwaving the details on polynomials):

<figure><img src="../.gitbook/assets/Untitled 4 (3).png" alt="" width="375"><figcaption></figcaption></figure>

Now, our model matches our data great! But intuitively, we can see that it won’t do very well predicting points other than those we measured - which was the whole point of our model.

Instead, when we use a slightly less complicated curve, we end up with something that looks a lot better( the true curve is marked in blue).

<figure><img src="../.gitbook/assets/Untitled 5 (3).png" alt="" width="375"><figcaption></figcaption></figure>

To put it simply, we need to find a good balance of telling our model how much `importance` should each data point have on our model. Too little and it mostly ignores the details and goe with the simplest trend, too much and it will try to incorporate even the noise as part of important details.

So how do we recognize if our models are fitted correctly? We use a _**validation set.**_ (In this simple 2D case we can easily get an intuitive feel for how well our model is doing, but in complicated functions such as those of a neural network are very complex.)

The validation set is **only** used to see how we are doing. Importantly, the model never gets to see and learn from the validation set. (hiding the validation set is akin to hiding the answers to a test; if the students knew the test answers it would just spit them out without learning any of the concepts in the class.)

Our Transformers use a `DatasetDict` for holding training and validation sets. We use `train_test_split` to create our validation sets, randomly picking 25% of the data to do so.

```python
dds = tok_ds.train_test_split(0.25, seed=42)
dds
### output ###
DatasetDict({
    train: Dataset({
        features: ['id', 'anchor', 'target', 'context', 'labels', 'input', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 27354
    })
    test: Dataset({
        features: ['id', 'anchor', 'target', 'context', 'labels', 'input', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 9119
    })
})
```

(In HuggingFace the **validation set** is called `test` and not `validate` , though the concept is similar.)

In practice, _how_ we should split our data depends on the task. Random splits can work, but not always the best. More info [here](https://www.fast.ai/posts/2017-11-13-validation-sets.html) on this pretty critical topic.

## Metrics and correlation

When we are training a model, we measure the performance of our models using **metrics**. These are measurements that we want to maximize which hopefully represent how well our model works for us. The whole topic of optimizing metrics is a huge, huge issue with a bunch of issues of practicality, accuracy, bias… the list goes on.

This isn’t new to AI, or any field for that matter; measuring how well _anything_ predicts the future is a remarkably difficult task. Here is a quote I read from a course I took by a Dr. Rachel Thomas:

> At their heart, what most current AI approaches do is to optimize metrics. The practice of optimizing metrics is not new nor unique to AI, yet AI can be particularly efficient (even too efficient!) at doing so. This is important to understand, because any risks of optimizing metrics are heightened by AI. While metrics can be useful in their proper place, there are harms when they are unthinkingly applied. Some of the scariest instances of algorithms run amok all result from over-emphasizing metrics. We have to understand this dynamic in order to understand the urgent risks we are facing due to misuse of AI.

With any metric, which is often a function, we often have a difficult time understanding what it does, For example, this is the [_Pearson correlation coefficient_](https://en.wikipedia.org/wiki/Pearson\_correlation\_coefficient)_:_

$$
r = \frac{\sum_{i=1}^n (x_i - \overline{x})(y_i - \overline{y})}{\sqrt{\sum_{i=1}^n (x_i - \overline{x})^2} \sqrt{\sum_{i=1}^n (y_i - \overline{y})^2}}
$$

Which probably means nothing to you unless you are a statistician. Instead, let’s try and understand the behavior. While value varies between -1 and 1 for perfect inverse correlation and perfect positive correlation, we still might not know what it does.

As an example, let's look at a subset of the [California Housing](https://scikit-learn.org/stable/datasets/real\_world.html#california-housing-dataset) dataset, which shows "_the median house value for California districts, expressed in hundreds of thousands of dollars_".

To visualize how our metric, represented as $$r$$ works, we can pick out a few data points between 2 variables. Lets look at `MedInc` and `MedHouseVal`, median income and median house value respectively.

<figure><img src="../.gitbook/assets/Untitled 7 (3).png" alt="" width="375"><figcaption></figcaption></figure>

Looks like there is a close relationship, but still a lot of variation. In this plot, the housing prices above 500k have been truncated to that maximum value.

Let’s look at Average Rooms instead:

<figure><img src="../.gitbook/assets/Untitled 8 (3).png" alt="" width="375"><figcaption></figcaption></figure>

The relationship looks somewhat like our previous graph, with higher incomes seemingly correlating to higher average rooms. While the graph looks a bit squished, this is just a consequence of the different scale of the graphs.

What’s interesting to note is that our $$r$$ is lower in this case. The reason is that there are a lot more outliers - values of `AveRooms` well outside the mean. Looking at the topmost data point, there is a building with 40 rooms, which means that it's probably a dorm or similar group housing. Comparing that against actual homes isn’t what we want, so lets remove this outlier among others (buildings with rooms above 15) and try again.

<figure><img src="../.gitbook/assets/Untitled 9 (3).png" alt="" width="375"><figcaption></figcaption></figure>

Now, we get a graph that looks much like our first comparison.

What does all this mean? For one, now we know that our coefficient is sensitive to outliers. Based on that knowledge, we can make more informed decisions, such as whether or not to remove our outliers or gain a better idea if this is the right metric we want to use. Getting a feel for a metric is always a good idea.

## Training the model

Most online materials at this point either do one of two things: (1) Either give you the code with little explanation and hand-wave the details, leaving you to fill in the blanks, or (2) Go into detail of how each and every component works, and the reason why it is the way it is. Because training is complicated, these are the easiest ways to teach the concept.

While (2) is great for the deeply invested, (1) leaves you clueless. I prefer a middle ground: explain the pseudocode and give just enough details for you to look into if you are interested.

The steps are relatively simple:

1. Format your data in a way the model understands,
2. pick good parameters(i.e tradeoffs) for your model (think “do I want my model to learn more slowly, or faster but less deeply?”)
3. Train the model
4. look at metrics and decide if it’s good enough, if not go back to either point (2) or use better data.

Here’s the code for training a model in HuggingFace using our earlier classification task, which was comparing 2 different words or short phrases, and scoring their similarity.

```python
from transformers import TrainingArguments,Trainer

bs = 128 #batch size, ie. how much data we want to give our model at once
epochs = 4 #how many times the model `practices` by looking at the input again
lr = 8e-5 #set how fast we want our model to learn

#Don't worry about details
#Just know we are creating the template or instructions for our model
args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=0.01, report_to='none')
  
#selecting the right task for our model and giving it our args    
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                  tokenizer=tokz, compute_metrics=corr_d)   
                  
#note we set compute_metrics to be our metric we want to evalutate, in this case our Pearson coefficient

#finally, we train our model
trainer.train();
```

One of the things our model returns is this table:

<figure><img src="../.gitbook/assets/Untitled 10 (4).png" alt=""><figcaption></figcaption></figure>

We can see our Pearson coefficient going down, which means our model is getting better!

note: Our Pearson correlation is calculated between the predicted and actual similarity, and we use it because it’s the most widely used to measure the degree of a relationship between 2 variables. For other problems, other metrics might be more suited.

The general gist of training a model is quite simple (at least made simple to us), but again all the underlying components have good reasons as to why they are built that way. While you definitely can and are encouraged to go deeper, I know that many people like me are impatient and want to skip over the details. In my limited experience, there are times where learning the details greatly help in my understanding of a concept. On the other hand, I sometimes feel like wasting my time when I consider my overall goals. There’s no right answer; you just have to pick where you put your time.

I hope this at least somewhat useful to anyone that happens to come across it. I’ll greatly appreciate any criticisms or comments.

This page is adapted from my own understanding of this Kaggle notebook:

{% embed url="https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners" %}
