class ParTree_node():

    def __init__(self, idx, label, is_leaf=False, clf=None, node_l=None, node_r=None, samples=None, support=None,
                 bic=None, is_oblique=None):
        self.idx = idx
        self.label = label
        self.is_leaf = is_leaf
        self.clf = clf
        self.node_l = node_l
        self.node_r = node_r
        self.samples = samples
        self.support = support
        self.bic = bic
        self.is_oblique = is_oblique
