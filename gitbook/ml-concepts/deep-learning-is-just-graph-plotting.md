# 📿 Deep learning is just graph plotting

The goal of a neural network is to learn the underlying function(or sets of patterns) that captures our data. We want to fit our function, such that it most accurately captures our data.

let's use a simpler example than a neural network:

<figure><img src="../.gitbook/assets/image (7).png" alt="" width="375"><figcaption></figcaption></figure>

This quadratic is of the form $$ax^2+bx+c$$. This will be the _underlying function_ that we generate our data from. In other words, this is the golden answer we would like to find out that is hidden from our 'neural network' . Next, let's add some noise, because in real life nothing is perfect.&#x20;

<figure><img src="../.gitbook/assets/image (8).png" alt="" width="375"><figcaption></figcaption></figure>



Our aim is for the 'neural network' to estimate the actual values of $$a, b, c$$, when we only know the general form of the function. One way we could try to estimate the values would be to plug in numbers randomly, taking all 3 to be $$1,1,1:$$

<figure><img src="../.gitbook/assets/image (9).png" alt="" width="375"><figcaption></figcaption></figure>

That looks pretty close. But how do we define '**close'**?

We want some way of knowing how close we are to the ‘best fit curve’ for our data. One metric we could use is the _**mean absolute error**_ or MAE,  which answers the question 'How far is each data point from this curve?

<figure><img src="../.gitbook/assets/image.png" alt="" width="375"><figcaption></figcaption></figure>



Our goal is to get the MAE, or more generally _**loss**_, to be as low as possible. 2.42 looks okay, but there is definitely a lot of room for improvement. But, doing this by hand by manually adjusting the values(i.e moving the curve around) is tedious enough for a quadratic equation in 2D. Imagine doing this for 3D, or even thousands of dimensions!

\
To start to automate this process, let’s first define our function simple quadratic function:\


```
def quad(a, b, c, x): 
 return a*x**2 + b*x + c
```

If we fix some values for a, b and c, we will have made a function. We can do that using the `partial` function:

```
def mk_quad(a,b,c): 
 return partial(quad, a,b,c)
```

And passing in the 3 values recreates our original quadratic.

```
f2 = mk_quad(3,2,1)
plot_function(f2)
```

<figure><img src="../.gitbook/assets/image (1).png" alt="" width="375"><figcaption></figcaption></figure>

Our function for Mean Absolute Error takes the data points of the graph we try to fit, our estimated curve, and calculates the mean.

```
def mae(preds, acts): 
 return (torch.abs(preds-acts)).mean()
```

Again, our basic idea is to minimize the loss, and “try to make the curves of the graph match our data”. But how do we tell our function to do that?

We can look at the gradient, or slope, of our function. We ask: by adjusting the values of our parameters, `a`, `b`, and `c`, how does it change our `mae`? (we partially differentiate with respect to each to get our answer.) If, say, `a` has a _negative_ gradient, then we know that increasing `a` will decrease `mae()` , and that we should increase `a`. This is because the gradient is a vector that points in the direction of the steepest ascent of the function. In simpler terms, it tells you how to change your parameters to increase the function's value the most.

To do this, first we need a function that takes all the parameters `a`, `b`, and `c` as a single vector input, and returns the value `mae()` applied to our quadratic curve based on those parameters.

```
def quad_mae(params):
    f = mk_quad(*params)
    return mae(f(x), y)
```

\
Let’s try it out:\


```
quad_mae([1.1, 1.1, 1.1])
# returns:
# tensor(2.4219, dtype=torch.float64)
```

note that our function returns us a _tensor._ for now, we can just think of tensors as arrays which can be 1, 2 or n-dimensional. Almost all functions in PyTorch work with tensors.

We arbitrarily pick 3 starting values for our function, and also to tell PyTorch that we want to calculate the gradients _with respect to each value_.

```
abc = torch.tensor([1.1, 1.1, 1.1], requires_grad=True)
```

`requires_grad` tells Python that we want to automatically update the gradient of this tensor, whenever it is used in a function.

```
loss = quad_mae(abc)
loss # thing we want to minimize, or so called 'inaccuracy'
# tensor(2.4219, dtype=torch.float64, grad_fn=<MeanBackward0>)
```

to actually calculate the loss, we call `backward()` . This updates our tensor `abc` , stored in an attribute `abc.grad`.

```
loss.backward()
abc.grad
#tensor([-1.3529, -0.0316, -0.5000])
```

From gradients, it seems like all our parameters are all low(and negative). Lets adjust the actual values by subtracting(because they are all negative) the gradient multiplied by a **small number**:

```python
with torch.no_grad():
    abc -= abc.grad*0.01
    loss = quad_mae(abc)
    
print(f'loss={loss:.2f}')
# loss=2.40
```

Our loss has gone down a little!

the **small number** is called the _learning rate,_ and is the most important hyperparameter that we will set when training a neural network. It is a measure of how much we want to ‘adjust’ the function each time.

note that the line `torch.no_grad()` is basically telling python _**not**_ to re-calculate the gradient of `abc` at this step, because this isn’t our loss function, and we don’t want python to update the gradient as we are just updating the values.

we can loop through this a couple more times:

```python
for i in range(10):
    loss = quad_mae(abc)
    loss.backward()
    with torch.no_grad(): abc -= abc.grad*0.01
    print(f'step={i}; loss={loss:.2f}')
```

```
step=0; loss=2.40
step=1; loss=2.36
step=2; loss=2.30
step=3; loss=2.21
step=4; loss=2.11
step=5; loss=1.98
step=6; loss=1.85
step=7; loss=1.72
step=8; loss=1.58
step=9; loss=1.46
```

and we can see that our loss is decreasing steadily.

This whole concept is called _**gradient descent**_, and this is the whole foundation of deep learning.

But why don’t we just keep looping this until the value stops decreasing? why are we setting our learning rate so low when we could decrease our loss faster?

Lets represent our loss against the value of our parameter.

<figure><img src="../.gitbook/assets/image (2).png" alt="" width="375"><figcaption></figcaption></figure>

A small learning rate means that the model will have to take more steps towards finding the minimum; which means the model takes longer to train.



But if we have a high learning rate, our loss can actually get worse!

<figure><img src="../.gitbook/assets/image (3).png" alt="" width="375"><figcaption></figcaption></figure>



Suppose then you set the learning rate to be as high as possible, while decreasing. This might lead to your model ‘bouncing’ around, which again increases the time to train the model.

<figure><img src="../.gitbook/assets/image (4).png" alt="" width="375"><figcaption></figcaption></figure>





In later sections, I’ll go into more detail when I talk about the commonly used methods for gradient descent, such as SGD(Stochastic Gradient Descent) or the ADAM optimizer.

But let’s go back to our main example. It turns out that you can boil down many, many problems to fitting a curve to a line. At it’s core, a neural network is just one single mathematical function with one special property: [_**infinitely expressive**_](https://en.wikipedia.org/wiki/Universal\_approximation\_theorem)_**.**_ A neural network can approximate _**any**_ function that is computable. And those functions can be creating images, writing essays, diagnosing diseases, almost anything.

\
To see why our curve fitting example is relevant, lets look at something even simpler: The function\
_𝑚𝑎𝑥_(_𝑥_,0), which just replaces all the negative values with zero.

<figure><img src="../.gitbook/assets/image (5).png" alt="" width="375"><figcaption></figcaption></figure>



This isn’t very interesting. And for now, the question of why we replace negative values with zero can be thought of as predicting the brightness of a pixel in an image. There isn’t a need to predict a value lower than 0 brightness, so we set it to zero. This is called the _rectified linear unit,_ _or (ReLU)._

Again, in the first image, changing the values of our linear function $$mx + b$$ changes the slope and 'position' of our graph. But look at what happens once we add two of our ReLUs, with different $$f(x) = x * e^{2 pi i \xi x}$$﻿$$a$$

and $$b$$ values together:

<figure><img src="../.gitbook/assets/image (6).png" alt=""><figcaption></figcaption></figure>

We can start to see something like our initial quadratic curve. By adding more and more ReLUs, or other non-linear functions together, we can eventually approximate any function, across not just two but any number of dimensions.&#x20;
