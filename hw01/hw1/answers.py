r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
'''
Increasing k will not lead to improve generalization, in general.
For the extermal value 1 - each sample will be classified according to the one closest sample in the training set, in this case there will be errors because the closest sample in the training set doens't always have the correct prediction.
For the extermal value of the size of the training set, each sample will be labled according to all the samples in the training set, therefore each sample will be classified the same, in this case it is clear we will have many errors.
Therefore, we would like to increase k up to the point it gives enough options for the data to be labeled, but not to get too far away from the closest sample.
'''
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""

The selection of the delta specifies the length of the margin, but we can notice that both delta and lambda control the tradeoff between the data loss and the regularization. therefore, by choosing different delta's we can also choose different lamba's which will give us the same results for those different delta's - meaning we can choose an arbitrary delta.
"""

part3_q2 = r"""

1. We can see in the images that the numbers are written in yellow and the background is green. therefore, it seems that the model checks weather the yellow area is parallel to the white area in the picture (the number itself) and the green area is parallel to the black (the background).
In images that were labeled incorrectly the number in the image (the white area) was more parallel to the yellow in the incorrect class that was predicted that in the correct class - and the same with the background.
2. It is similar in the way that in both KNN and here we are trying to find the closest image from k (here 9) images.
It is different because in KNN we predict according to the closest k training and in SVM we predict according to the highest dot product score.
"""

part3_q3 = r"""

1. I would sat that the learning rate is good. the lost descends gradualy and uniformly. for a too high rate we would see it misses the minimum of the loss function and has too big steps. for too low we would see if descends too slowly and takes too long to reach the minimum.
2. Slightly overfitted - the training accuracy is very high and the loss is higher for the validation.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""

The ideal pattern is where all points reside on the x-axis as it indicates zero validation error.
In the case of our model, the points are scattered around the x-axis in a symmetical matter, indicating that there's no bias towards one side.
The vast majority of the points are in close proximity to the x-axis and therefore we can infer that the model's fitness is high.

In the top-5 features plot, the points are scattered unevenly around the x-axis as the points that are the furthest from it tend to be below it.
The many far points indicate a less fit model.
"""

part4_q2 = r"""
1. We used logspace instead of linspace because 𝜆 is a regularizaion constant, in particular a multiplicative constant. 
   Therefore, its impact on the results is much more subtle, so it requires a great change in value to inccur a 
   noticable change in the model's accuracy. Thereby logspace is a fitting scale.
   (Notice how in the degree's case, a very slight change can "make or break" the model's performance)
2. 180 ( = k_folds * |degree_range| * |𝜆_range|)

"""

# ==============
