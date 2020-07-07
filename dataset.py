import torch.utils.data as data
import numpy as np

class RecSys_Dataset(data.Dataset):

    def __init__(self, features, num_item, train_mat=None, num_negative=0, is_training=None):
        super(RecSys_Dataset, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.features_positive = features           # positive pair (user, item)
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_negative = num_negative
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    def negative_sample(self):
        assert self.is_training, 'No need to sampling when testing'

        self.features_negative = []         # negative pair
        for x in self.features_positive:
            u = x[0]
            # For each user u in a positive pair, sample num_negative pair (u, j) | (u, j) not in train_mat
            for t in range(self.num_negative):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_negative.append([u, j])

        labels_positive = [1 for _ in range(len(self.features_positive))]       # positive pair have label = 1
        labels_negative = [0 for _ in range(len(self.features_negative))]       # negative pair have label = 0

        self.features_fill = self.features_positive + self.features_negative    # list all pair
        self.labels_fill = labels_positive + labels_negative                    # lust all labels correspond to list all pair

    def __len__(self):
        # total pair
        return (self.num_negative + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training else self.features_positive
        labels = self.labels_fill if self.is_training else self.labels
        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label