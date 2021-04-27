# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import create_dataset
import evaluate_metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

# myEvaluation = evaluate_metrics.Evaluation(
#     dir1="../starting_project/cad_samples/finished")
myEvaluation = evaluate_metrics.Evaluation(
    dir1="../starting_project/samples")
myEvaluation.creation_from_csv("./orestis_rawData.csv")
