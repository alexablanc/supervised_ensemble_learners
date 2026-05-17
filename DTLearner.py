import numpy as np

def make_leaf(y):
    node = np.empty((1,4), dtype=float)
    node[0,0] = -1
    node[0,1] = y
    node[0,2] = np.nan
    node[0,3] = np.nan
    return node

def make_root(feature_idx, split_val, left_rows):
    node = np.empty((1,4), dtype=float)
    node[0,0] = feature_idx
    node[0,1] = split_val
    node[0,2] = 1
    node[0,3] = left_rows + 1
    return node

class DTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.invalid_splits = 0
        self.total_splits = 0
        self.verbose = verbose
        self.leaf_size = leaf_size
        self.tree = None

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "ablanc6"  # replace tb34 with your Georgia Tech username

    def study_group(self):
        return "ablanc6"

    def add_evidence(self, train_x, train_y):
        self.tree = self.build_tree(train_x, train_y)

    def build_tree(self, train_x, train_y):
        np.seterr(divide='ignore', invalid='ignore')
        #print("Building tree", train_x.shape[0])
        if train_x.shape[0] <= self.leaf_size:
            #print("nan count: ", np.isnan(train_x).sum())
            #return np.array([[-1,np.nanmean(train_y),np.nan,np.nan]])
            return make_leaf(np.nanmean(train_y))
        if np.all(train_y == train_y[0]):
            #return np.array([[-1,train_y[0],np.nan,np.nan]])
            return make_leaf(train_y[0])
        else:
            # calculate correlations between factors and y values and take the factor with the highest correlation
            ycol = train_y.reshape(-1,1)
            stacked = np.hstack((train_x, ycol))
            corrs = np.corrcoef(stacked, rowvar=False)
            corrs = np.abs(corrs[-1, :-1])
            corrs = np.nan_to_num(corrs, nan=0.0)
            if np.max(corrs) == 0:
                #return np.array([[-1,np.nanmean(train_y),np.nan,np.nan]])
                return make_leaf(np.nanmean(train_y))
            best_factor = int(np.argmax(corrs))
        splitval = np.nanmedian(train_x[:, best_factor])
        if np.isnan(splitval):
            #return np.array([[-1,np.nanmean(train_y),np.nan,np.nan]])
            return make_leaf(np.nanmean(train_y))
        left_vals = train_x[:, best_factor] <= splitval
        right_vals = train_x[:, best_factor] > splitval
        left_count = np.sum(left_vals)
        right_count = np.sum(right_vals)
        if left_count == 0 or right_count == 0:
            #return np.array([[-1,np.nanmean(train_y),np.nan,np.nan]])
            return make_leaf(np.nanmean(train_y))
        lefttree = self.build_tree(train_x[left_vals], train_y[left_vals])
        righttree = self.build_tree(train_x[right_vals], train_y[right_vals])
        #root = np.array([[best_factor, splitval, 1, lefttree.shape[0] + 1]])
        root = make_root(best_factor, splitval, lefttree.shape[0])
        return np.vstack((root, lefttree, righttree))

    def query(self, test_x):
        pred = np.zeros(test_x.shape[0])
        for i in range(test_x.shape[0]):
            node = 0
            while True:
                feature = int(self.tree[node,0])
                #print("TOP:","i", i, "node", node, "feature", feature)
                #print("node: ", node, "row: ", self.tree[node])
                if feature == -1:
                    #print("HIT LEAF at node", node, "pred", self.tree[node,1])
                    pred[i] = self.tree[node,1]
                    break
                splitval = self.tree[node,1]
                #xval = test_x[i, feature]
                #print("split check:", "xval",xval, "<= splitval", splitval)
                if test_x[i, feature] <= splitval:
                    #print("GO LEFT offset", self.tree[node, 2])
                    node = node + int(self.tree[node,2])

                else:
                    #print("GO RIGHT offset", self.tree[node, 3])
                    node = node + int(self.tree[node,3])
        return pred

