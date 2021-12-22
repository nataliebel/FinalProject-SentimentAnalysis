import csv
import json
import random
from Models.dictionaryClassifier import get_tweets_weights_feature
from support.Utils import get_json_tweet_list
from datetime import datetime
import pandas as pd
import numpy as np

TEST_FILE = r"C:\SentimentAnalysisProject\Models\Data\test set for the bootstrapper.json"
TRAIN_FILE = r"C:\SentimentAnalysisProject\Models\Data\train-set for the bootstrapper.json"
STOP_WORDS = r"C:\SentimentAnalysisProject\Models\Data\heb_stop_words.txt"
TWEETS_CSV_FILE = r"C:\SentimentAnalysisProject\Models\Data\train.csv"


def save_results(df, nm, is_trans):
    """
    saves results for test session in csv file
    :param df: data frame contains the predictions
    :param nm: model name
    :param is_trans: bool, is data translated
    """
    # get current date
    datetime_object = datetime.now()
    dt = datetime_object.strftime("%d_%m_%H_%M")
    fname = "results/" + nm + "_results_" + dt
    if is_trans:
        fname += "_tr"
    df.to_csv(fname + ".csv")


def save_file(file_name, features, ids, polarity, subjectivity):
    df = pd.DataFrame({"ids":ids, "features":np.array(features),"polarity": np.array(polarity), "subjectivity":np.array(subjectivity)})
    df.to_csv(file_name+".csv", index=False, mode='a', header=False, encoding='utf8')


def get_tweets(pos_f, neg_f):
    """
    deserialize tweets for test session
    :param pos_f: name of the positive tweets file
    :param neg_f: name of the negative tweets file
    :return: positive and negative tweets
    """
    with open(pos_f, 'r', encoding="utf-8") as pos_json_file:
        pos_d = json.load(pos_json_file)
    with open(neg_f, 'r', encoding="utf-8") as neg_json_file:
        neg_d = json.load(neg_json_file)
    positive_tweets = pos_d['tweets']
    negative_tweets = neg_d['tweets']
    for itemP in positive_tweets:
        itemP.update({"label": 1})

    for itemN in negative_tweets:
        itemN.update({"label": -1})

    return positive_tweets, negative_tweets


def get_train_test_tweets():
    train_set = get_json_tweet_list(TRAIN_FILE)
    test_set = get_json_tweet_list(TEST_FILE)

    return train_set, test_set


def get_test_set():
    test_set = get_json_tweet_list(TEST_FILE)
    return test_set


def get_vocabulary():
    """
    deserialize vocabulary file
    :return: list of words
    """
    with open("positive-words.txt", 'r') as file:
        vocabulary = file.read().split('\n')[:1000]
    with open("negative-words.txt", 'r') as file:
        vocabulary += file.read().split('\n')[:1000]
    return list(set(dict.fromkeys(vocabulary)))


def separate_data(data, language='hebrew'):
    """
    separate data to features and labels
    :param language:
    :param data: original data
    :return: separated data
    """
    polarity = None
    subjectivity = None
    features = []
    try:
        df_data = pd.DataFrame(data)
        if 'label' in df_data.columns:
            df_data['label'] = df_data['label'].astype(str)
            df_data['polarity'] = df_data['label'].str.slice(15, 17)
            df_data.polarity = np.where(df_data['polarity'].str.contains('\''),
                                        df_data['polarity'].str.slice(1, 2), df_data['polarity'].str.slice(0, 1))
            df_data.polarity = df_data.polarity.astype(int)
            df_data['subjectivity'] = df_data['label'].str.slice(41, -2)
            df_data['is_topic'] = df_data['subjectivity'] == 'topic'
            df_data.is_topic = df_data.is_topic.astype(int)
            polarity = list(df_data.polarity)
            subjectivity = list(df_data.is_topic)
        ids = df_data.iloc[:, 2].values
        lan = 'input'
        if language == 'english':
            lan = 'translatedText'

        for _, item in df_data.iterrows():
            if type(item['extended_tweet']) is not float:
                if type(item['extended_tweet']['full_text']) is list:
                    # if item['extended_tweet']['full_text']._len_() == 0:
                    #     ids.Remove(item['id_str'])
                    features.append(item['extended_tweet']['full_text'][0][lan])
                else:
                    features.append(item['extended_tweet']['full_text'])
            else:
                if type(item['text']) is list:
                    # if item['text']._len_() == 0:
                    #     ids.Remove(item['id_str'])
                    features.append(item['text'][0][lan])
                else:
                    features.append(item['text'])
        return ids, features, polarity, subjectivity
    except:
        print("can't separate data")


def extract_stop_words():
    """
    deserialize vocabulary file
    :return: list of words
    """
    with open(STOP_WORDS, 'r', encoding="utf-8") as file:
        vocabulary = file.read().split('\n')
        return vocabulary


def clean_data(data):
    ids = data[:, 0]
    features = data[:, 1]
    polarity = data[:, 2].astype(int)
    subjectivity = data[:, 3].astype(int)
    return ids, features, polarity, subjectivity


def get_tweets_from_csv():
    df = pd.read_csv(filepath_or_buffer=TWEETS_CSV_FILE)
    train_size = int(df.__len__() * 0.8)
    random.shuffle(df.values)
    train = df.loc[:train_size,:]
    test = df.loc[train_size:,:]
    return train, test


def add_dictionary_feature(ids, data, polarity, lan='hebrew'):
    df = pd.DataFrame({'ids': ids, 'tweet_words': data, 'label': polarity})
    df = get_tweets_weights_feature(df, language=lan)
    return df['vocab_feature']


def calc_avg(param):
    return str(((param[0] + param[1] + param[2] + param[3] + param[4])/5)*100)


def check_values_acc(predictions, filtered_test_set, polarity):
    right = 0
    almost_right = 0
    if polarity is True:
        for real_pol, predict_pol in zip(filtered_test_set[2], predictions[1]):
            if real_pol == predict_pol:
                right += 1
            if real_pol == predict_pol + 1 or real_pol == predict_pol - 1 or real_pol == predict_pol:
                almost_right += 1
        print("Polarity: The real accuracy in this iteration -> " + str(right/len(predictions[1])))
        print("Polarity: The almost real accuracy in this iteration -> " + str(almost_right / len(predictions[1])))
    else:
        for real_subject, predict_subject in zip(filtered_test_set[3], predictions[1]):
            if real_subject == predict_subject:
                right += 1
            if predict_subject !=0 and predict_subject!=1:
                print(predict_subject)
        print("Subjectivity: The real Subjectivity in this iteration -> " + str(right / len(predictions[1])))


# TODO
def remove_zeros(train_ids, filtered_data_train, polarity_Y, subjectivity_Y, zero_index_list):
    objects_to_convert = list()
    objects_to_convert.extend([train_ids, filtered_data_train, polarity_Y, subjectivity_Y])

    for obj, i in zip(objects_to_convert, range(len(objects_to_convert))):
        if type(obj) is list:
            objects_to_convert[i] = [i for j, i in enumerate(obj) if j not in zero_index_list]
            # for index in zero_index_list:
            #     obj.remove(index)
        elif type(obj) is np.ndarray:
            for index in zero_index_list:
                objects_to_convert[i] = np.delete(obj, index)
        # else:
        #     print("wrong objects to remove the zeros")

    return train_ids, filtered_data_train, polarity_Y, subjectivity_Y
