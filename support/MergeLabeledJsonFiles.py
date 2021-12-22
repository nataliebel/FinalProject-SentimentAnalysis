"""
This script is charge for calculate and create new JSON with the final and average numbers
This takes all the JSONs stored and checks the average of the labels
"""
import sys

from support.JsonManager import JsonManager
from support.Utils import script_opener, json_files_collector, marge_all_json_file, check_duplicate_tweet, \
    dir_checker_creator, retweet_checker, get_json_tweet_list, create_json_dict_file, check_json_format

sys.path.append(r'C:\Users\dembo\Documents\Computer Science\Third Year\Project\Sentiment Analysis Project')

LABELED_JSONS_TEMP = 'Temp files/backup/old_labelers_action'
LABELED_JSONS = 'Temp files/labeled files'
UNLABELED_JSON = 'Temp files/unlabeled JSON'


def average_tweet_calc(tweet, positivity, relativity):
    """
    calculates the real average of the label
    :param tweet: the tweet we want to work with
    :param positivity: number
    :param relativity: number
    :return: the tweet with the new labels value
    """
    number_of_labelers = tweet["JSON-Manager"]["labelers"]
    previous_label_positivity = tweet["label"]["positivity"]
    previous_label_relativity = tweet["label"]["relative subject"]

    # calc the value of the labels that where before
    prev_val_positivity = previous_label_positivity * number_of_labelers
    prev_val_relativity = previous_label_relativity * number_of_labelers

    # writes the new average label number
    tweet["JSON-Manager"]["labelers"] = number_of_labelers + 1
    tweet["label"]["positivity"] = (prev_val_positivity + positivity) / (number_of_labelers + 1)
    tweet["label"]["relative subject"] = (prev_val_relativity + relativity) / (number_of_labelers + 1)

    return tweet


def change_tweet_relative_label(tweet_to_refactor):
    """
    changes the format from name label to number
    :param tweet_to_refactor: some tweet. this is a dictionary
    :return:
    """
    try:
        tweet_to_refactor["label"]["positivity"] = float(tweet_to_refactor["label"]["positivity"])
        if tweet_to_refactor["label"]["relative subject"] == "person":
            tweet_to_refactor["label"]["relative subject"] = 1.0
        else:
            tweet_to_refactor["label"]["relative subject"] = 0.0
        tweet_to_refactor["JSON-Manager"] = {"labelers": 1,
                                             "relative-number": 0.0}
        return tweet_to_refactor
    except:
        # print("this is unlabeled tweet") TODO: change the exception to condition
        return None


def refactor_labels_back(new_fixed_tweets):
    """
    refactor the value back to the model using format
    :param new_fixed_tweets: the final list
    :return: the final list with the refactor values
    """
    for tweet in new_fixed_tweets:
        tweet["label"]["positivity"] = int(tweet["label"]["positivity"])
        if tweet["label"]["relative subject"] >= 0.5:
            tweet["label"]["relative subject"] = "person"
        else:
            tweet["label"]["relative subject"] = "topic"
    return new_fixed_tweets


def main_json_merger():
    """
    runs the merger of multi jsons to one list
    :return:
    """
    new_fixed_json = list()
    # for every tweet check by the id the sum of all the next tweets
    for tweet in initial_merged_json:
        checking_id = tweet["id"]
        # during the loop it refactors the label format for average sum purpose
        fixed_tweet = change_tweet_relative_label(initial_merged_json[0])
        del initial_merged_json[0]

        if fixed_tweet is None:
            continue

        # next for every tweet we calculate the average with all the suitable tweets with the same id use
        for cmp_tweet, i in zip(initial_merged_json, range(initial_merged_json.__len__())):
            if cmp_tweet["id"] != checking_id:
                continue

            cmp_tweet = change_tweet_relative_label(cmp_tweet)
            if cmp_tweet is None:
                del initial_merged_json[i]
                continue

            # calculate the average
            fixed_tweet = average_tweet_calc(fixed_tweet, cmp_tweet["label"]["positivity"],
                                             cmp_tweet["label"]["relative subject"])
            del initial_merged_json[i]

        # adds the new tweet to the final list
        new_fixed_json.append(fixed_tweet)

    new_fixed_json = refactor_labels_back(new_fixed_json)
    return new_fixed_json


def labeled_and_unlabeled_json_creator():
    """
    creates two list of tweets. One of the labeled tweets and one to the unlabeled tweets separately
    :return:
    """

    # creates the main labeled list
    print("\nCollecting labeled tweets...")
    json_files_list = json_files_collector(path=LABELED_JSONS)
    initial_merged_json = marge_all_json_file(file_list=json_files_list)
    labeled_total_list = check_duplicate_tweet(initial_merged_json)
    labeled_total_list = check_json_format(json_list=labeled_total_list)

    # creates the main unlabeled list
    print("\nCollecting unlabeled tweets...")
    json_files_list = json_files_collector(path=UNLABELED_JSON)
    initial_merged_json = marge_all_json_file(file_list=json_files_list)
    unlabeled_total_list = retweet_checker(check_duplicate_tweet(initial_merged_json))

    manager = JsonManager(unlabeled_total_list)
    manager.remove_double_tweets(labeled_total_list)
    manager.save_new_json_manager_file(list_to_be_saved=labeled_total_list, name='new total labeled')
    # saves new no labeled tweets
    manager.save_new_json_manager_file(list_to_be_saved=manager.create_json_with_quotes(), name='no-labeled-tweets')
    manager.summarize_labeled_tweets(json_to_summarize=labeled_total_list)


def change_labels_value(new_tweet_label_values):
    """
    when we do merge we must check the values are the correct value type
    :param new_tweet_label_values:
    :return:
    """
    counter = 0
    change_to = get_json_tweet_list(src_json_file=new_tweet_label_values)
    candidate_files = json_files_collector(path=LABELED_JSONS)
    for f in candidate_files:
        json_file_to_change = get_json_tweet_list(f)
        for t in json_file_to_change:
            for change_t in change_to:
                if change_t['id'] == t['id'] and t["label"]["positivity"] != change_t["label"]["positivity"]:
                    old = t["label"]["positivity"]
                    t["label"]["positivity"] = change_t["label"]["positivity"]
                    print(str(old) + " changed to -> " + str(change_t["label"]["positivity"]))
                    counter += 1
        create_json_dict_file(json_list=json_file_to_change, json_file_name=f)
    print(str(counter) + " tweets changed\n")


if __name__ == '__main__':
    script_opener("JSON merger")
    new_json = list()

    user_choose = input("\nPlease choose your action:\nFor regular file merge - press 1\n"
                        "For creating new labeled and unlabeled JSONs - press 2\n"
                        "For revalue the labeled tweets - press 3\n")

    if user_choose == '1':
        # gathering all the tweets together
        json_files_list = json_files_collector(path=LABELED_JSONS_TEMP)
        initial_merged_json = marge_all_json_file(file_list=json_files_list)
        new_json = main_json_merger()

    elif user_choose == '2':
        labeled_and_unlabeled_json_creator()
        print("New unlabeled and labeled file has been created\n")

    # this option for changing the value of labels
    elif user_choose == '3':
        change_labels_value(new_tweet_label_values="3_labeled_tweets.json")

    else:
        print("bad input")

    print('DONE - bye')

