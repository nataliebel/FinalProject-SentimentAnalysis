import random

from Models.Model import *
from support.Utils import create_json_dict_file, get_json_tweet_list, script_opener
from Models import model_utils as mUtils
import nltk

IS_STEMMED = False
TEST_RATIO = 10
VALIDATION_CONST = 0.7
MANUAL_LABELING_FILE = "../Models/Data/manual_labeling.json"
TRAIN_FILE = mUtils.TRAIN_FILE


class Bootstrapper(object):
    """
    bootstrapper constructor
    :param model: model type
    :param train_set: train set for training the model - data and labels
    :param data_test: data for labeling
    """
    def __init__(self, model, train_set, data_test):
        self.my_model = model
        if IS_STEMMED:
            self.none_labeled_tweets = data_test.values
            self.model_data_set = train_set.values
        else:
            self.none_labeled_tweets = data_test
            self.model_data_set = train_set
        self.final_data = list()
        self.manual_labeling = get_json_tweet_list(MANUAL_LABELING_FILE)
        self.is_loaded = True


    def execute(self):
        """
        runs a while loop - runs the model and gets predictions and adds the predicted data
        (with high probability) to the data_set - for training the model in the future.
        the loop runs until the test_set is empty.
        """
        random.shuffle(self.none_labeled_tweets)
        i = 1
        while self.none_labeled_tweets is not None and self.none_labeled_tweets.__len__() > 0:
            print("\nstart of execute number -> " + str(i))
            print("tweets left: " + str(len(self.none_labeled_tweets)))
            random.shuffle(self.model_data_set)
            self.my_model_test_tweets = self.get_test_tweets()
            model_results, confidence, sub_results, sub_confidence = \
            self.my_model.run(self.model_data_set, self.my_model_test_tweets, self.is_loaded, IS_STEMMED)
            self.validate_model_solution(model_results, confidence, sub_results, sub_confidence)
            print("\nend of execute number -> " + str(i) + "\n")
            i += 1
        self.save_new_train_set()
        self.my_model.save_models()
        self.my_model.save_vectorizer()

    def get_test_tweets(self):
        """
        each time the function return a slice of the test_set
        :return: test set for current iteration
        """
        self.ratio = int(len(self.model_data_set)*(TEST_RATIO/100))
        test_tweets = self.none_labeled_tweets

        if len(test_tweets) > self.ratio:
            test_tweets = test_tweets[: self.ratio]
            self.none_labeled_tweets = self.none_labeled_tweets[self.ratio: -1]
        else:
            test_tweets = test_tweets[: -1]
            self.none_labeled_tweets = None
        return test_tweets

    def validate_model_solution(self, results, confidence, sub_results, sub_confidence):
        """
        validates prediction for each example in data test
        :param results: polarity predictions
        :param confidence: probability of polarity prediction
        :param sub_results: subjectivity predictions
        :param sub_confidence: probability of subjectivity predictions
        """
        good_res_tweets = 0
        for id, conf, res, sub_conf, sub_res in zip(results[0], confidence, results[1], sub_confidence, sub_results[1]):
            if conf[res-1] >= VALIDATION_CONST/5 and sub_conf[sub_res] >= VALIDATION_CONST:
                good_res_tweets += 1
                self.append_to_train_set(id, res, sub_res)
            else:
                self.manual_labeling.append(self.find_by_id(id))
        print(str(good_res_tweets) + " was added to the train set")

    def append_to_train_set(self, id, result, sub_res):
        """
        append a new example from test set to train set - if it's probability is high enough
        :param id: example id
        :param result: example polarity prediction
        :param sub_res: example subjectivity prediction
        """
        tweet = self.find_by_id(id)
        tweet["label"] = {"positivity": str(result), "relative subject": str(sub_res)}
        self.final_data.append(tweet)
        self.model_data_set.append(tweet)

    def find_by_id(self, id):
        """
        finds the tweet with the given id in test set
        :param id: id of the wanted tweet
        :return: wanted tweet
        """
        tweet = next(t for t in self.my_model_test_tweets if t["id_str"] == id)
        return tweet

    def save_new_train_set(self):
        """
        serialize the new train set - combined with old train set and data from test set
        """
        create_json_dict_file(self.model_data_set, TRAIN_FILE)
        create_json_dict_file(self.manual_labeling, MANUAL_LABELING_FILE)


if __name__ == '__main__':
    script_opener(script_title="Bootstrapper")
    if IS_STEMMED:
        train, test = mUtils.get_tweets_from_csv()
    else:
        train, test = mUtils.get_train_test_tweets()
    stop_words = mUtils.extract_stop_words()
    model = Model(stop_words, language='english')
    bootStrapper = Bootstrapper(model, train, test)
    bootStrapper.execute()

