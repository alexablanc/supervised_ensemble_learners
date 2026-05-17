import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False) for l in range(20)]
    def author(self):
        return "ablanc6"  # replace tb34 with your Georgia Tech username
    def add_evidence(self, train_x, train_y):
        for learner in self.learners:
            learner.add_evidence(train_x, train_y)
    def query(self, test_x):
        points = np.array([learner.query(test_x) for learner in self.learners])
        return points.mean(axis=0)