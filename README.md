Machine Learning Algorithm for differentiating dogs from cats written in Python3 using CNN with Keras
To run the program type python3 dog_or_cat.py
Make sure to have the dataset linked below and use the proper folder structure

Dependencies:
- numpy
- matplotlib.pyplot
- keras
- tensorflow
- sklearn


I divided them like this:
3000 cat pictures and 3000 dog pictures in the test folder
7999 cat pictures and 7999 dog pictures in the training folder
1500 cat pictures and 1500 dog pictures in the validation folder

The folder structure should look like this:
test_folder:
    test:
        0.jpg
        ...
        n.jpg
training:
    training_cat:
        0.jpg
        ...
        n.jpg
    training_dog:
        0.jpg
        ...
        n.jpg
validation:
    val_cat:
        0.jpg
        ...
        n.jpg
    val_dog:
        0.jpg
        ...
        n.jpg

I've used Keras' built in methods for early stopping so it will run for either 30 epochs or until it has 3 epochs in a row with no performance increase to the validation set.
I could've added more layers and set the image size larger than 64x64, but running machine learning on a CPU takes a fair amount of time and I wanted to make it runnable.

As you can see from Example_run.png I ended at approximately 83% on the validation set and the printed matrix was:
TP 2232, FP 768, FN 819, TN 2191. It gives us about 66% success rate.
