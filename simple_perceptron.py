
"""

To run, copy and paste into your terminal:
python .\simple_perceptron.py


To test:
Change values of INPUTS and WEIGHTS

"""

# Step Function Threshold
THRESHOLD = 1.5

INPUTS = [1, 0, 1, 0, 1]
# Every input should have a corresponding weight.
WEIGHTS = [0.7, 0.6, 0.5, 0.3, 0.4]


# Perceptron Calculation
sum = 0

for i in range(len(INPUTS)):
    sum += INPUTS[i] * WEIGHTS[i]

# Add bias
BIAS = 0.2
sum += BIAS

# Step Function activation. 1 is True, 0 is False.
if sum > THRESHOLD:
    print("Output: 1")
else:
    print("Output: 0")


