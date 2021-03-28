""" Evaluate VowelModel

This short program is used to evaluate a trained model of type VowelModel.
The model is trained for predicting vowels based on their surronding characters.

Provided the model to evaluate, a test text, and an output filename,
the program calculates and prints the accuracy of the model and writes
a new text file based on the test text with the predicted vowels.

This code is a joint effort by:
    Sigrid Jonsson
    Sarab Youssef
    Judit Casademont Moner

"""
import sys
import argparse
import torch
import numpy as np
from model import VowelModel

vowels = sorted(['y', 'é', 'ö', 'a', 'i', 'å', 'u', 'ä', 'e', 'o'])


def tokenize_text(filepath):
    """Tokenize text from filepath into list of characters

    Args:
        filepath (string): Path to text to be tokenized

    Returns:
        tuple: Complete tokenized text and a set of each distinct character
    """
    tokenized = []
    with open(filepath, "r") as textfile:
        for line in textfile:
            tokenized += [char for char in line]

    tokenized = ["<s>", "<s>"] + tokenized + ["<e>", "<e>"]
    return tokenized, list(set(tokenized))


def create_array(feat, vocab):
    """Create feature array

    Creates a zeroed np array the length of vocab. If feat is included in vocab,
    then set its corresponding index to one.

    Args:
        feat (string): Feature to be set
        vocab (list): List of distinct characters; all possible features

    Returns:
        np.array: Feature array with feat element set to one.
    """
    feature_array = np.zeros(len(vocab))
    if feat in vocab:
        feature_array[vocab.index(feat)] = 1
    return feature_array


def create_matrix(tokenized_text, vocab):
    """Create feature matrix for all vowels in tokenized_text

    Args:
        tokenized_text (list): List of characters to extract features from
        vocab (list): List of distinct characters; all possible features

    Returns:
        tuple: A feature matrix and a list of corresponding vowel indices.
    """
    vowel_idx_list = []
    feature_matrix = []
    for idx in range(len(tokenized_text) - 4):
        vowel_idx = idx + 2
        if tokenized_text[vowel_idx] not in vowels:
            continue
        vowel_idx = vowels.index(tokenized_text[vowel_idx])
        vowel_idx_list.append(vowel_idx)

        features = [tokenized_text[idx], tokenized_text[idx+1],
                    tokenized_text[idx+3], tokenized_text[idx+4]]
        feature_array = np.concatenate(
            [create_array(feat, vocab) for feat in features])
        feature_matrix.append(feature_array)

    return np.array(feature_matrix), np.array(vowel_idx_list)


def print_accuracy(expected, predicted):
    """Calculates and prints accuracy

    Args:
        expected (list): List of the expected elements, ground truth
        predicted (list): List of predicted elements
    """
    correct = 0
    for exp, pred in zip(expected, predicted):
        if exp == pred:
            correct += 1

    print('Accuracy: %d %%' % (100 * correct / len(expected)))


def convert_to_vowels(vowel_indices):
    """Converts vowel_indices to actual vowel characters

    Args:
        vowel_indices (list): List of indices

    Returns:
        list: List of vowels
    """
    vowels_list = []
    for vowel_index in vowel_indices:
        vowels_list.append(vowels[vowel_index])
    return vowels_list


def write_predicted_text(origintext, predicted, outputfile):
    """Replace vowels in original text to predicted and write to file

    Args:
        origintext (list): The original text in tokenized form
        predicted (list): List of predicted vowels
        outputfile (string): Path to output file
    """
    predicted_text = []
    for character in origintext:
        if character in vowels:
            predicted_text.append(predicted.pop(0))
        elif character in ['<s>', '<e>']:
            pass
        else:
            predicted_text.append(character)

    with open(outputfile, 'w') as outputfile:
        outputfile.write(''.join(predicted_text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modelpath", type=str)
    parser.add_argument("testpath", type=str)
    parser.add_argument("outputfile", type=str)
    args = parser.parse_args()

    # Load the model
    model = torch.load(args.modelpath)
    model.eval()

    # Prepare the test data
    tokenized_data = tokenize_text(args.testpath)
    unrecognized_chars = []
    test_matrix, expected = create_matrix(tokenized_data[0], model.vocab)

    # Run test data through model and get predictions
    with torch.no_grad():
        outputs = model(torch.Tensor(test_matrix))
        # Get the predicted vowel indices
        _, predicted = torch.max(outputs.data, 1)

    print_accuracy(expected, predicted)
    predicted_vowles = convert_to_vowels(predicted)
    write_predicted_text(tokenized_data[0], predicted_vowles, args.outputfile)
