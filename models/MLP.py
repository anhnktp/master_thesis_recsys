import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, num_user, num_item, num_factor, num_layer_mlp, dropout, MLP_model=None):
        super(MLP, self).__init__()
        """
        num_user: number of users;
        num_item: number of items;
        factor_num: number of predictive factors or latent factors;
        num_layer_mlp: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        MLP_model: pre-trained MLP weights.
        """
        self.dropout = dropout
        self.MLP_model = MLP_model
        self.embed_user_MLP = nn.Embedding(
            num_embeddings=num_user, embedding_dim=num_factor * (2 ** (num_layer_mlp - 1)))
        self.embed_item_MLP = nn.Embedding(
            num_embeddings=num_item, embedding_dim=num_factor * (2 ** (num_layer_mlp - 1)))

        self.MLP_layers = nn.Sequential()
        for i in range(num_layer_mlp):
            input_size = num_factor * (2 ** (num_layer_mlp - i))
            self.MLP_layers.add_module('linear%d' %i, nn.Linear(in_features=input_size, out_features=input_size // 2))
            self.MLP_layers.add_module('relu%d' %i, nn.ReLU())
            self.MLP_layers.add_module('dropout%d' %i, nn.Dropout(p=self.dropout))

        self.predict_layer = nn.Linear(in_features=num_factor, out_features=1)

        # initialize weight
        self._init_weight_()

    def _init_weight_(self):
        """
            Initialize weights by normal distribution N(mean=0.0, std=0.01) or load from pretrained model
        """
        if not self.MLP_model:
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.01)
            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_MLP.weight = self.MLP_model.embed_user_MLP.weight
            self.embed_item_MLP.weight = self.MLP_model.embed_item_MLP.weight
            # mlp layers
            for (m1, m2) in zip(
                    self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
            # predict layers
            self.predict_layer.weight, self.predict_layer.bias = self.MLP_model.predict_layer.weight, self.MLP_model.predict_layer.bias

    def forward(self, user, item):
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)
        prediction = self.predict_layer(output_MLP).view(-1)
        return prediction