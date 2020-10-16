import torch.nn as nn
from activations import Mish

class GMF(nn.Module):

    def __init__(self, num_user, num_item, num_factor, GMF_model=None):
        super(GMF, self).__init__()
        """
        num_user: number of users;
        num_item: number of items;
        factor_num: number of predictive factors or latent factors;
        GMF_model: pre-trained GMF weights;
        """
        self.embed_user_GMF = nn.Embedding(num_embeddings=num_user, embedding_dim=num_factor)
        self.embed_item_GMF = nn.Embedding(num_embeddings=num_item, embedding_dim=num_factor)

        self.predict_layer = nn.Linear(in_features=num_factor, out_features=1)

        # initialize weight
        self._init_weight_(GMF_model)

    def _init_weight_(self, GMF_model=None):
        """
            Initialize weights by normal distribution N(mean=0.0, std=0.01) or load from pretrained model
        """
        if not GMF_model:
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.01)
            nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight = GMF_model.embed_user_GMF.weight
            self.embed_item_GMF.weight = GMF_model.embed_item_GMF.weight
            # predict layers
            self.predict_layer.weight, self.predict_layer.bias = GMF_model.predict_layer.weight, GMF_model.predict_layer.bias

    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF
        prediction = self.predict_layer(output_GMF).view(-1)
        return prediction