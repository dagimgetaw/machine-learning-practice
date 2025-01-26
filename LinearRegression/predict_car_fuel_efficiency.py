import numpy as np
from weight_and_bias_calc import slope, bias, weights,mpg
from math import fabs, pow

user_input = int(input("Enter the weight of the car in pound: "))
user_input_float = float(user_input / 1000)
result = slope * user_input_float + bias

if user_input_float in weights:
    index = int(np.where(weights == user_input_float)[0][0])
    label = mpg[index]
    loss = result - label
    print(f"The loss is in MAE is {fabs(loss)}")
    print(f"The loss is in MSE is {pow(loss, 2)}")
    print(mpg[index])
    print("+", index)

print(f"The predicted fuel efficiency is {result} miles per gallon")
