import torch
import torch.nn as nn
import torch.nn.functional as F


class SEC2NCF(nn.Module):

    def __init__(self, num_user, num_item, num_factor, dropout, model_type, GMF_model=None):
        super(SEC2NCF, self).__init__()
        """
        num_user: number of users;
        num_item: number of items;
        factor_num: number of predictive factors or latent factors;
        num_layer_mlp: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model_type: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights.
        """
        self.dropout = dropout
        self.model = model_type
        self.GMF_model = GMF_model
        self.embed_user_GMF = nn.Embedding(num_embeddings=num_user, embedding_dim=num_factor)
        self.embed_item_GMF = nn.Embedding(num_embeddings=num_item, embedding_dim=num_factor)
        self.conv1 = nn.Conv2d(1, num_factor, (2,2))

        self.predict_layer = nn.Linear(in_features=num_factor*(num_factor - 1), out_features=1)

        # initialize weight
        self._init_weight_()

    def _init_weight_(self):
        """
            Initialize weights by normal distribution N(mean=0.0, std=0.01) or load from pretrained model
        """
        if 'pre' not in self.model:
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.conv1.weight, std=0.01)
            nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)


    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF
        stack = torch.stack([embed_user_GMF, embed_item_GMF], dim=-2)
        stack = torch.unsqueeze(stack, dim=1)
        output_SEC = F.relu(self.conv1(stack))
        output_SEC = output_SEC.view(output_SEC.size()[0], -1)
        prediction = self.predict_layer(output_SEC).view(-1)
        return prediction.view(-1)

