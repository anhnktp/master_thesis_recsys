import torch
import torch.nn as nn
from activations import Mish



class SeConvNCF(nn.Module):

    def __init__(self, num_user, num_item, num_factor, dropout, num_fm, GMF_model=None, MLP_model=None):
        super(SeConvNCF, self).__init__()
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
        self.embed_user_01 = nn.Embedding(num_embeddings=num_user, embedding_dim=num_factor)
        self.embed_item_01 = nn.Embedding(num_embeddings=num_item, embedding_dim=num_factor)
        self.embed_user_02 = nn.Embedding(num_embeddings=num_user, embedding_dim=num_factor)
        self.embed_item_02 = nn.Embedding(num_embeddings=num_item, embedding_dim=num_factor)

        self.SeNCF_modules = nn.Sequential()
        self.SeNCF_modules.add_module('conv1', nn.Conv2d(1, num_fm, (2, 2), stride=2, padding=1))
        self.SeNCF_modules.add_module('batchnorm1', nn.BatchNorm2d(num_fm))
        self.SeNCF_modules.add_module('relu1', nn.ReLU(inplace=True))

        self.SeNCF_modules.add_module('conv2', nn.Conv2d(num_fm, num_fm, (2, 2), stride=1, padding=0))
        self.SeNCF_modules.add_module('batchnorm2', nn.BatchNorm2d(num_fm))
        self.SeNCF_modules.add_module('relu2', nn.ReLU(inplace=True))

        self.SeNCF_modules.add_module('conv3', nn.Conv2d(num_fm, 1, (1, 1), stride=1, padding=0))
        self.SeNCF_modules.add_module('batchnorm3', nn.BatchNorm2d(1))
        self.SeNCF_modules.add_module('relu3', nn.ReLU(inplace=True))


        self.ConvNCF_modules = nn.Sequential()
        self.ConvNCF_modules.add_module('conv1', nn.Conv2d(1, num_fm, (2, 2), stride=2))
        self.ConvNCF_modules.add_module('batchnorm1', nn.BatchNorm2d(num_fm))
        self.ConvNCF_modules.add_module('relu1', nn.ReLU(inplace=True))

        num_conv_layers = 1
        size = int(num_factor // 2)
        while size != 1:
            num_conv_layers += 1
            self.ConvNCF_modules.add_module('conv%d' % num_conv_layers, nn.Conv2d(num_fm, num_fm, (2, 2), stride=2))
            self.ConvNCF_modules.add_module('batchnorm%d' % num_conv_layers, nn.BatchNorm2d(num_fm))
            self.ConvNCF_modules.add_module('relu%d' % num_conv_layers, nn.ReLU())
            size = int(size // 2)

        self.predict_layer = nn.Linear(in_features=num_fm + int(num_factor // 2), out_features=1)

        # initialize weight
        self._init_weight_(GMF_model, MLP_model)

    def _init_weight_(self, GMF_model=None, MLP_model=None):
        """
            Initialize weights by normal distribution N(mean=0.0, std=0.01) or load from pretrained model
        """
        if not GMF_model:
            nn.init.normal_(self.embed_user.weight, std=0.01)
            nn.init.normal_(self.embed_item.weight, std=0.01)
        else:
            # embedding layers
            self.embed_user_01.weight.data.copy_(GMF_model.embed_user_GMF.weight)
            self.embed_item_01.weight.data.copy_(GMF_model.embed_item_GMF.weight)

        if not MLP_model:
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        else:
            self.embed_user_02.weight.data.copy_(MLP_model.embed_user_MLP.weight)
            self.embed_item_02.weight.data.copy_(MLP_model.embed_item_MLP.weight)

        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
        for m in self.SeNCF_modules.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

        for m in self.ConvNCF_modules.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()


    def forward(self, user, item):

        embed_user_01 = self.embed_user_01(user)
        embed_item_01 = self.embed_item_01(item)
        stack = torch.stack([embed_user_01, embed_item_01], dim=-2)
        stack = torch.unsqueeze(stack, dim=1)
        output_SeNCF = self.SeNCF_modules(stack)
        output_SeNCF = output_SeNCF.view(output_SeNCF.size()[0], -1)

        embed_user_02 = self.embed_user_02(user)
        embed_item_02 = self.embed_item_02(item)
        outer_product = torch.einsum('mi,mj->mij', [embed_user_02, embed_item_02])
        outer_product = torch.unsqueeze(outer_product, dim=1)
        output_ConvNCF = self.ConvNCF_modules(outer_product)
        output_ConvNCF = output_ConvNCF.view(output_ConvNCF.size()[0], -1)
        concat = torch.cat((output_SeNCF, output_ConvNCF), -1)

        prediction = self.predict_layer(concat).view(-1)

        return prediction

