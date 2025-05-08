# OR problem setup
# concept of logical OR
sample_data = [[0,0],
               [0,1],
               [1,0],
               [1,1]]

expected_results = [0,1,1,1]
activation_threshold = 0.5

from random import random
import numpy as np

weights = np.random.random(2)/1000 # Small random float 0 < w < .001
weights 

bias_weight = np.random.random() / 1000
bias_weight

# Perceptron random guessing
for idx, sample in enumerate(sample_data):
    input_vector = np.array(sample)
    activation_level = np.dot(input_vector, weights) + (bias_weight * 1)
    if activation_level > activation_threshold:
        perceptron_output = 1
    else:
        perceptron_output = 0
        print(f'Predicted: {perceptron_output}')
        print(f'Expected: {expected_results[idx]}')
        print()


# Perceptron learning
for iternation_num in range(5):
    correct_answers = 0
    for idx, sample in enumerate(sample_data):
        input_vector = np.array(sample)
        weights = np.array(weights)
        activation_level = np.dot(input_vector, weights) + (bias_weight*1)
        if activation_level > activation_threshold:
            perceptron_output = 1
        else:
            perceptron_output = 0
        if perceptron_output == expected_results[idx]:
            correct_answers += 1
        new_weights = []
        for i, x in enumerate(sample):
            new_weights.append(weights[i] + (expected_results[idx] - perceptron_output) * x)
        bias_weight = bias_weight + ((expected_results[idx] - perceptron_output) * 1)
        weights = np.array(new_weights)
    print(f'{correct_answers} correct answers out of 5, for iteration {iternation_num}')