import pandas as pd
import time
import numpy as np

def filter_by_timestamp(csv_df, query_timestamp, info):
    '''

    :param csv_df: pandas dataframe
           query_timestamp: unix timestamp
    :return: csv_df panda dataframe after filter by csv_df['timestamp] <= query_timestamp and csv_df['timestamp] == info
            and drop duplicate row with same 'shopper ID' and only keep row with nearest unix timestamp with query_timestamp
    '''
    res = csv_df.loc[(csv_df['timestamp'] <= query_timestamp) & (csv_df['info'] == info)]
    return res.drop_duplicates(subset=['shopper ID'], keep='last')

def convert_to_unix_time(human_time):
    # return time.mktime(time.strptime(human_time, '%Y-%m-%d %H:%M:%S.%f'))
    return time.strptime(human_time, '%Y-%m-%d %H:%M:%S.%f')

def load_csv(path, col=None):
    return pd.read_csv(path, usecols=col).rename(columns={'timestamp (unix timestamp)': 'timestamp'})
if __name__ == '__main__':
    # csv_360_path = '/Users/anhvu/PycharmProjects/YOLOv3_TensorFlow/log_person_tracking.csv'
    csv_attention_path = '/Users/anhvu/PycharmProjects/YOLOv3_TensorFlow/log_signage_attention2.csv'
    # csv_360_df = pd.read_csv(csv_360_path).rename(columns={'timestamp (unix timestamp)': 'timestamp'})
    # print(csv_360_df.head())
    # # csv_attention_df = pd.read_csv(csv_attention_path)
    csv_attention_df = pd.read_csv(csv_attention_path, parse_dates=[-3], date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    # csv_attention_df = csv_attention_df.rename(columns={'Timestamp (UTC-JST)': 'timestamp'})
    print(csv_attention_df.head())
    # csv_attention_df['timestamp'].apply(lambda x: convert_to_unix_time(x))
    csv_attention_df['timestamp'] = csv_attention_df['Timestamp (UTC-JST)'].values.astype(np.int64) / 10 ** 9 - 7*3600
    # csv_attention_df.drop(columns=['Shopper_ID'], inplace=True)
    print(csv_attention_df.head())
    csv_attention_df.to_csv(csv_attention_path, index=False)
    #
    # query_timestamp = 1586946943
    # data_360 = csv_360_df.loc[(csv_360_df['timestamp'] <= query_timestamp) & (csv_360_df['info'] == 'A')]
    # data_360 = data_360.drop_duplicates(subset=['shopper ID'], keep='last')
    # csv_touch_path = '/Users/anhvu/PycharmProjects/YOLOv3_TensorFlow/log_shelf_touch.csv'
    # csv_shelf_touch = load_csv(csv_touch_path, col=['shelf ID', 'hand_coords', 'timestamp (unix timestamp)'])
    # print(len(csv_shelf_touch))
    # i = 0
    # anchor = 1586946225
    # while csv_shelf_touch['timestamp'][i] < anchor: i += 1
    # print(csv_shelf_touch['shelf ID'])
