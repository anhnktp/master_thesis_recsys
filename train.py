import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from dataset import RecSys_Dataset
import config
from models import *
from utils import *
from tqdm import trange
import evaluate

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data', help='root path of datasets folder')
    parser.add_argument('--model_root', type=str, default='./checkpoints', help='checkpoints directory')
    parser.add_argument('--dataset_name', type=str, default='ml-1m', choices=['ml-1m', 'pinterest'],
                        help='root path of datasets folder')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--model_type', type=str, default='GMF',
                        choices=['GMF', 'MLP', 'NeuMF', 'Se2NCF', 'Se3NCF', 'SeConvNCF', 'DNCF', 'ConvNCF'],
                        help='model type')
    parser.add_argument('--mf_pretrain', type=str, default='',
                        help='Specify the pretrain model filename for GMF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', type=str, default='',
                        help='Specify the pretrain model filename for MLP part. If empty, no pretrain will be used')
    parser.add_argument('--learner', type=str, default='adam',
                        choices=['adam', 'adagrad', 'rmsprop', 'sgd', 'adadelta', 'adasparse', 'asgd', 'adamax', 'adamw'],
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--lr_scheduler', default=False, action='store_true',
                        help='whether to use  Cyclic learning rate scheduler during training')
    parser.add_argument('--freeze', default=False, action='store_true',
                        help='whether to freeze pre-trained model except prediction layer')
    parser.add_argument('--l2reg', type=float, default=0.,
                        help='l2 regularization')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size in training')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epoches')
    parser.add_argument('--top_k', type=int, default=10, help='compute metric@top_k')
    parser.add_argument('--num_factor', type=int, default=64, help='predictive factors numbers in the model')
    parser.add_argument('--num_fm', type=int, default=64, help='number of feature map used in CNN layers in the model')
    parser.add_argument('--num_layer_mlp', type=int, default=3, help='number of layers in MLP model')
    parser.add_argument('--num_neg', type=int, default=4, help='sample negative items for training')
    parser.add_argument('--test_num_neg', type=int, default=99, help='sample part of negative items for testing')
    parser.add_argument('--out', type=bool, default=True, help='whether to save model')
    parser.add_argument('--device', type=str, default='0', help='gpu card id or cpu')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    print('Config: ', args)
    model_dir = args.model_root
    learner = args.learner
    use_lrs = args.lr_scheduler
    freeze = args.freeze
    resultsdfpath = os.path.join(model_dir, 'results_df.p')
    if use_lrs: tag_lrs = 'wlrs'
    else: tag_lrs = 'wolrs'

    GMF_model_path = args.mf_pretrain
    MLP_model_path = args.mlp_pretrain

    if osp.isfile(GMF_model_path):
        GMF_model = torch.load(GMF_model_path)
        tag_pretrain = 'with_pretrain'
    else:
        GMF_model = None
        tag_pretrain = 'no_pretrain'
    if osp.isfile(MLP_model_path):
        MLP_model = torch.load(MLP_model_path)
        tag_pretrain = 'with_pretrain'
    else:
        MLP_model = None


    exp_name = '{}_{}_lr{}_{}_{}_{}factor'.format(args.model_type, tag_pretrain, args.lr, learner, tag_lrs, args.num_factor)
    train_data, test_data, num_user, num_item, train_mat = load_data(test_num=100, data_root=args.data_root, dataset_name=args.dataset_name)
    train_dataset = RecSys_Dataset(train_data, num_item, train_mat, args.num_neg, True)
    test_dataset = RecSys_Dataset(test_data, num_item, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=args.test_num_neg + 1, shuffle=False, num_workers=0)


    if args.model_type == 'GMF':
        model = GMF(num_user, num_item, args.num_factor, GMF_model)
    elif args.model_type == 'MLP':
        model = MLP(num_user, num_item, args.num_factor, args.num_layer_mlp, args.dropout, MLP_model)
    elif args.model_type == 'NeuMF':
        model = NeuMF(num_user, num_item, args.num_factor, args.num_layer_mlp,
                              args.dropout, GMF_model, MLP_model)
    elif args.model_type == 'Se2NCF':
        model = Se2NCF(num_user, num_item, args.num_factor, args.dropout, args.num_fm, GMF_model)
    elif args.model_type == 'Se3NCF':
        model = Se3NCF(num_user, num_item, args.num_factor, args.dropout, args.num_fm, GMF_model, MLP_model)
    elif args.model_type == 'SeConvNCF':
        model = SeConvNCF(num_user, num_item, args.num_factor, args.dropout, args.num_fm, GMF_model, MLP_model)
    elif args.model_type == 'DNCF':
        model = DNCF(num_user, num_item, args.num_factor, args.num_layer_mlp, args.dropout, GMF_model, MLP_model)
    elif args.model_type == 'ConvNCF':
        model = ConvNCF(num_user, num_item, args.num_factor, args.dropout, GMF_model)

    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    print('######### Model architecture #########')
    print(model)
    model = model.cuda()
    if freeze:
        for name, layer in model.named_parameters():
            if ("embed" in name):
                layer.requires_grad = False

    loss_function = nn.BCEWithLogitsLoss()
    if args.learner == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2reg)
    elif args.learner == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2reg, nesterov=True)
    elif args.learner == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2reg)
    elif args.learner == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.l2reg)
    elif args.learner == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.l2reg)
    elif args.learner == 'adasparse':
        optimizer = optim.SparseAdam(model.parameters(), lr=args.lr)
    elif args.learner == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.l2reg)
    elif args.learner == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2reg)
    elif args.learner == 'asgd':
        optimizer = optim.ASGD(model.parameters(), lr=args.lr, weight_decay=args.l2reg)
    training_steps = len(train_loader)

    step_size = training_steps * 3  # one cycle every 6 epochs
    cycle_momentum = False
    if learner == 'sgd' or learner == 'rmsprop':
        cycle_momentum = True
    if use_lrs:
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1 * args.lr, max_lr=args.lr, cycle_momentum=cycle_momentum)
    else:
        scheduler = None

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable_params of {} model: {}'.format(args.model_type, trainable_params))
    writer = SummaryWriter(log_dir='runs/{}'.format(exp_name))  # for visualization

    ########################### TRAINING #####################################
    best_hr = 0
    config.engine_logger.critical('Starting training ...')
    step_id = 0
    for epoch in range(args.epochs):
        model.train()  # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.negative_sample()
        total_loss = 0
        with trange(training_steps) as t:
            for user, item, label in train_loader:
                t.set_description('Epoch %i' % epoch)
                # model.train()
                user = user.cuda()
                item = item.cuda()
                label = label.float().cuda()

                model.zero_grad()
                prediction = model(user, item)
                loss = loss_function(prediction, label)
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                    lr = scheduler.get_lr()[0]
                else: lr = args.lr
                total_loss += loss.item()
                t.set_postfix(loss=np.sqrt(total_loss / (step_id + 1)), lr=lr)

                # model.eval()
                avg_loss = total_loss / training_steps
                writer.add_scalar('data/loss', avg_loss, step_id)
                step_id += 1
                t.update(1)
        HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
        writer.add_scalar('performance/HR@{}'.format(args.top_k), HR, epoch)
        writer.add_scalar('performance/NDCG@{}'.format(args.top_k), NDCG, epoch)
        if HR > best_hr:
            best_hr, best_ndcg, best_epoch, best_loss = HR, NDCG, epoch, avg_loss
            if args.out:
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                if not os.path.exists(exp_name):
                    os.makedirs(osp.join(model_dir, exp_name), exist_ok=True)
                torch.save(model,
                           osp.join(model_dir, exp_name, 'step_{}_HR_{}_NDCG_{}.pth'.format(step_id, HR, NDCG)))

        config.engine_logger.critical(
            "Metric| HR: {:.3f}\tNDCG: {:.3f}\tAvgLoss: {:.3f}".format(np.mean(HR), np.mean(NDCG), avg_loss))

    config.engine_logger.critical("----- End| Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f} -----".format( best_epoch, best_hr, best_ndcg))

    cols = ["exp_name", "epoch_loss", "best_hr", "best_ndcg", "best_epoch"]
    vals = [exp_name, best_loss, best_hr, best_ndcg, best_epoch]
    if not os.path.isfile(resultsdfpath):
        results_df = pd.DataFrame(columns=cols)
        experiment_df = pd.DataFrame(data=[vals], columns=cols)
        results_df = results_df.append(experiment_df, ignore_index=True)
        results_df.to_pickle(resultsdfpath)
    else:
        results_df = pd.read_pickle(resultsdfpath)
        experiment_df = pd.DataFrame(data=[vals], columns=cols)
        results_df = results_df.append(experiment_df, ignore_index=True)
        results_df.to_pickle(resultsdfpath)


