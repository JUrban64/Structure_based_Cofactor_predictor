import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import evaluate_model
from model_E3 import GraphClassifierE3

# Monkey-patch model again
evaluate_model.train.GraphClassifier = GraphClassifierE3
evaluate_model.GraphClassifier = GraphClassifierE3

if __name__ == '__main__':
    # Uživatel pravděpodobně používá evaluate_E3.py častějš než test script.
    evaluate_model.main()
