import numpy as np
import pandas as pd
import time
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import pickle
import os.path as osp
from joblib import Parallel, delayed

def load_data(test_num=100, data_root='./data', dataset_name='ml-1m'):

    # Config path to data files
    train_rating = osp.join(data_root, dataset_name, '{}.train.rating'.format(dataset_name))
    test_rating = osp.join(data_root, dataset_name, '{}.test.rating'.format(dataset_name))
    test_negative = osp.join(data_root, dataset_name, '{}.test.negative'.format(dataset_name))

    # Read data file
    train_data = pd.read_csv(train_rating, sep='\t', header=None, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    num_user = train_data['user'].max() + 1
    num_item = train_data['item'].max() + 1
    train_data = train_data.values.tolist()

    # load train ratings as a dok matrix
    train_mat = sp.dok_matrix((num_user, num_item), dtype=np.float32)
    for user_id, item_id in train_data:
        train_mat[user_id, item_id] = 1.0

    # Load test data
    test_data = []
    with open(test_negative, 'r') as f:
        line = f.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = f.readline()
    return train_data, test_data, num_user, num_item, train_mat

def array2mtx(interactions):
    num_users = interactions[:,0].max()
    num_items = interactions[:,1].max()
    mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
    for user, item, rating in interactions:
            mat[user, item] = rating
    return mat.tocsr()


def standard_split(df, data_path):
	# Cardinality
	n_users = df.user.nunique()
	n_items = df.item.nunique()
	n_ranks = df['rank'].nunique()
	train, test, = train_test_split(df.values.astype(np.int64), test_size=0.2, random_state=1)

	# Save
	np.savez(data_path/"standard_split.npz", train=train, test=test, n_users=n_users,
		n_items=n_items, n_ranks=n_ranks, columns=df.columns.tolist())


def neuralcf_split(df, data_path):
	# Xiangnan He, et al, 2017 train/test split with implicit negative feedback

	# sort by rank
	dfc = df.copy()

	# Cardinality
	n_users = df.user.nunique()
	n_items = df.item.nunique()

	dfc.sort_values(['user','rank'], ascending=[True,True], inplace=True)
	dfc.reset_index(inplace=True, drop=True)

	# use last ratings for testing and all the previous for training
	test = dfc.groupby('user').tail(1)
	train = pd.merge(dfc, test, on=['user','item'],
		how='outer', suffixes=('', '_y'))
	train = train[train.rating_y.isnull()]
	test = test[['user','item','rating']]
	train = train[['user','item','rating']]

	# select 99 random movies per user that were never rated by that user
	all_items = dfc.item.unique()
	rated_items = (dfc.groupby("user")['item']
	    .apply(list)
	    .reset_index()
	    ).item.tolist()

	def sample_not_rated(item_list, rseed=1, n=99):
		np.random.seed=rseed
		return np.random.choice(np.setdiff1d(all_items, item_list), n)

	print("sampling not rated items...")
	start = time()
	non_rated_items = Parallel(n_jobs=4)(delayed(sample_not_rated)(ri) for ri in rated_items)
	end = time() - start
	print("sampling took {} min".format(round(end/60,2)))

	negative = pd.DataFrame({'negative':non_rated_items})
	negative[['item_n'+str(i) for i in range(99)]] =\
		pd.DataFrame(negative.negative.values.tolist(), index= negative.index)
	negative.drop('negative', axis=1, inplace=True)
	negative = negative.stack().reset_index()
	negative = negative.iloc[:, [0,2]]
	negative.columns = ['user','item']
	negative['rating'] = 0
	assert negative.shape[0] == len(non_rated_items)*99
	test_negative = (pd.concat([test,negative])
		.sort_values('user', ascending=True)
		.reset_index(drop=True)
		)
	# Ensuring that the 1st element every 100 is the rated item. This is
	# fundamental for testing
	test_negative.sort_values(['user', 'rating'], ascending=[True,False], inplace=True)
	assert np.all(test_negative.values[0::100][:,2] != 0)

	# Save
	np.savez(data_path/"neuralcf_split.npz", train=train.values, test=test.values,
		test_negative=test_negative.values, negatives=np.array(non_rated_items),
		n_users=n_users, n_items=n_items)

	# Save training as sparse matrix
	print("saving training set as sparse matrix...")
	train_mtx = array2mtx(train.values)
	sp.save_npz(data_path/"neuralcf_train_sparse.npz", train_mtx)

def prepare_amazon(data_path, input_fname):

	df = pd.read_json(data_path/input_fname, lines=True)

	keep_cols = ['reviewerID', 'asin', 'unixReviewTime', 'overall']
	new_colnames = ['user', 'item', 'timestamp', 'rating']
	df = df[keep_cols]
	df.columns = new_colnames

	# rank of items bought
	df['rank'] = df.groupby("user")["timestamp"].rank(ascending=True, method='dense')
	df.drop("timestamp", axis=1, inplace=True)

	# mapping user and item ids to integers
	user_mappings = {k:v for v,k in enumerate(df.user.unique())}
	item_mappings = {k:v for v,k in enumerate(df.item.unique())}
	df['user'] = df['user'].map(user_mappings)
	df['item'] = df['item'].map(item_mappings)
	df = df[['user','item','rank','rating']].astype(np.int64)

	pickle.dump(user_mappings, open(data_path/'user_mappings.p', 'wb'))
	pickle.dump(item_mappings, open(data_path/'item_mappings.p', 'wb'))

	standard_split(df, data_path)
	neuralcf_split(df, data_path)

def convert_timestamp_to_human_time(unix_timestamp):
	return time.strftime("%d_%m_%Y_%H_%M_%S", time.localtime(unix_timestamp))