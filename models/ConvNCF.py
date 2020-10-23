import torch
import torch.nn as nn


class ConvNCF(nn.Module):

    def __init__(self, num_user, num_item, num_factor, dropout, GMF_model):
        super(ConvNCF, self).__init__()
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
        self.embed_user = nn.Embedding(num_embeddings=num_user, embedding_dim=num_factor)
        self.embed_item = nn.Embedding(num_embeddings=num_item, embedding_dim=num_factor)
        num_fm = int(num_factor // 2)
        self.ConvNCF_modules = nn.Sequential()
        self.ConvNCF_modules.add_module('conv1', nn.Conv2d(1, num_fm, (2,2), stride=2))
        self.ConvNCF_modules.add_module('batchnorm1', nn.BatchNorm2d(num_fm))
        self.ConvNCF_modules.add_module('relu1', nn.ReLU(inplace=True))

        num_conv_layers = 1
        size = num_fm
        while size != 1:
            num_conv_layers += 1
            self.ConvNCF_modules.add_module('conv%d' % num_conv_layers, nn.Conv2d(num_fm, num_fm, (2,2), stride=2))
            self.ConvNCF_modules.add_module('batchnorm%d' % num_conv_layers, nn.BatchNorm2d(num_fm))
            self.ConvNCF_modules.add_module('relu%d' % num_conv_layers, nn.ReLU())
            size = int(size // 2)

        self.predict_layer = nn.Linear(in_features=num_fm, out_features=1)
        self.logit = nn.Sigmoid()

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
        else:
            # embedding layers
            self.embed_user.weight.data.copy_(GMF_model.embed_user_GMF.weight)
            self.embed_item.weight.data.copy_(GMF_model.embed_item_GMF.weight)

        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
        for m in self.ConvNCF_modules.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()


    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        outer_product = torch.einsum('mi,mj->mij', [embed_user, embed_item])
        outer_product = torch.unsqueeze(outer_product, dim=1)
        output_ConvNCF = self.ConvNCF_modules(outer_product)
        output_ConvNCF = output_ConvNCF.view(output_ConvNCF.size()[0], -1)
        prediction = self.predict_layer(output_ConvNCF).view(-1)
        return prediction