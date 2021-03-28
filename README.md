# LT2222 V21 Assignment 3

Name: Sigrid Jonsson

## Part 1
a - Tokenizes the provided text into characters. Returns a the text as a tokenized characters list, and also a set of all characters in the list.

b - Creates a feature matrix for each vowel in the tokenized text

Traverse the list of tokenized characters.
If a vowel is found, convert it to a vowel index and add it to a list.
Pick the four characters surrounding the vowel; these are the vowels features.
Pass each of the feature characters to g, which returns a feature list. The four list combined completes the vowels feature list. Add it as a row to the feature matrix. Also converts the vowel to an index and add it to a corresponding list.
Returns both matrix and vowel indices list.

g -
Creates an array the size of the vocab and adds 1 for the feature.

Command line arguments:
* 'k' is hidden size of the neural network
* 'r' is the number of epochs to train the model
* 'm' is the input file, the text to train the model on
* 'h' is the file name/path the model is saved to


## Part 2
To run the test data through the trained model, the test data needed to be converted in the same fashion as the training data. For this, the initial idea was to import the 'a' and 'b' functions from 'train.py', to first tokenize the data and then build a feature matrix. There was a problem though: the training data and test data doesn't have the exact vocab. This wouldn't be too much of an issue, of the test data just had a smaller vocab than the training data, then we could just use the models vocab. However, the test data includes characters that are not included in the training data. To circumvent the issue, we decided to add a check when creating the matrix, to skip any character that is not included in the models vocab. Since we didn't want to add changes to 'train.py', we rewrote the functions we need to 'eval.py' instead.

After that, the evaluation is pretty straight forward: the newly created test matrix is run through the model and we get back a list of predicted vowel indices. With the predicted and expected vowel indices we calculate and print the accuracy. We also create a new text file with the actual vowels replaced with the predicted ones.

## Part 3

When training and testing models, I used the following parameters:

    r = default, k = [10, 100, 500, 1000, 2000]
    k = default, r = [10, 50, 200, 500, 1000]

### Accuracy scores:

| r/k | 10  | 100 | 500 | 1000 | 2000 |
|-----|-----|-----|-----|------|------|
| 100 | 10% | 33% | 31% | 47%<sup>*</sup> | 22%  |

<sup>*</sup> This value is most likely a lucky fluke. When retraining the model with the same parameters, the accuracy score is 13%.

| k/r | 10  | 50  | 200 | 500 | 1000 |
|-----|-----|-----|-----|-----|------|
| 200 | 23% | 34% | 34% | 10% | 12%  |

The general feel I get from this result, is as we say in Swedish: "Lagom är bäst", or "There is virtue in moderation". The best performing models are the ones with parameter values closer to the default (except k=1000, see above). A too small hidden size, 'k', performs very badly, but a too big size will also perform worse than the moderate default. The same goes for the number of epochs, that performs worse the longer you let it train. I guess if you train the model for too many epochs, it will become over fitted for the training data, and won't perform well when encountering the new test data.
A quick glance of the produced texts, one of the worst performing models, k=10, has replaced the vowels with just a few different characters, 'a', 'o', and 'å', where 'å' seems to occur in the 99% of the cases. The other worst performing model, r=500 have done a similar prediction, but with only 'å', 'i', and 'y', where 'y' is the most prominent. The best performing model, k=1000, has predicted a somewhat readable text. 

## Other notes

The code in this project was co-authored with Sarab Youssef and Judit Casademont Moner.
