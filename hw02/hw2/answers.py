r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd=0.01
    lr=0.05
    reg=0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp(opt_name):
    wstd, lr, reg, = 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different hyperparameters for each optimizer.
    # ====== YOUR CODE: ======
    if opt_name == 'vanilla':
        wsrd = 0.1
        lr = 0.02
        reg = 0.001
    if opt_name == 'momentum':
        wsrd = 0.1
        lr = 0.005
        reg = 0.001
    if opt_name == 'rmsprop':
        wsrd = 0.01
        lr = 0.0001
        reg = 0.0009
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wsrd = 0.001
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
1. They match what I expected to see: I expected the train loss to be smaller with no dropout, and that the difference between train and test loss will be higher with no dropout - and indeed the test loss with no drop out is higher than with dropout which means that the difference between train and test loss is higher with no dropout.
2. the test loss was similar with both dropout, but the accuracy was higher with the low-dropout, meaning the low=dropout it better. we can also see that the training loss was higher with the high-dropout - the dropout was probably too high to learn enough from the training.
"""

part2_q2 = r"""
It is possible because the loss and the accuracy are measured by different factors - while the loss checks how close we are to predicting correctly, the accuracy checks the number of correct predictions, and it doesnt matter how close the prediction was to the correct one. meaning, we can get more correct predictions so the accuracy will increase but the incorrect predictions can be far enough from the correct so the loss will increase as well.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
