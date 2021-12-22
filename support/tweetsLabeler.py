import re

from support.JsonManager import JsonManager
from support.Utils import get_json_tweet_list, create_json_dict_file, separate_debug_print_big, send_report_by_email, \
    script_opener, separate_debug_print_small, dir_checker_creator, retweet_checker

TRANSLATED_JSON = 'gidi_trans_check.json'
BACKUP_RATIO = 2
NAME = 'Gidi'


def initialize_data():
    """
    initialize all the lists: the main tweets data, the labeled and those
    the user couldn't label
    :return: the 3 list mentioned above
    """
    print("Loading json's data...\n")
    # create the directories for the program
    dir_checker_creator("Temp files/Backup")

    # creates the 3 main tweet list and checks them.
    main_json_list = retweet_checker(get_json_tweet_list('Temp files/' + TRANSLATED_JSON))
    labeled_list = get_json_tweet_list('Temp files/labeled_tweets.json')
    problems_list = get_json_tweet_list('Temp files/problem_tweets.json')

    print("Loading completed!\nTotal tweets you has labeled: " + str(len(labeled_list)) +
          "\nTotal tweets you mark as problematic: " + str(len(problems_list)) +
          "\nTotal tweets you left the label: " + str(len(main_json_list)) +
          "\nPlease follow the instruction and good luck\n")

    return main_json_list, labeled_list, problems_list


def tweet_print(data_to_print, tweet_from_json):
    if 'extended_tweet' in tweet_from_json:
        data_to_print.append(tweet_from_json['extended_tweet']['full_text'][0])
        type_to_print = "Full text"
    else:
        data_to_print.append(tweet_from_json['text'][0])
        type_to_print = "Short text"
    return data_to_print, type_to_print


def print_tweet_data(cur_tweet, quoted=False):
    """
    print the data line by line
    :param quoted:
    :param cur_tweet: a single data tweet
    :return:
    """
    data = []

    if not isinstance(cur_tweet["text"], list):
        quoted = True

    if not quoted:
        # a simple tweet
        data, text_type = tweet_print(data, cur_tweet)
        print("This is simple tweet(" + text_type + "):")
        # prints the tweet's text
        try:
            for d in data[0]:
                print(d, ':', data[0][d])  # TODO: check about more fields
        except TypeError:
            print("It seems like you have json format issues\n"
                  "Please write down the tweet number -> {0}.\n"
                  "Exit the program now!\n".format(str(cur_tweet["id"])))
            finalize_json_data()
            exit(1)
    else:
        # a complicated tweet
        if 'extended_tweet' in cur_tweet:
            origin_text = cur_tweet['extended_tweet']['full_text']
            text_type = "Full text -"
        else:
            origin_text = cur_tweet['text']
            text_type = "Short text -"
        print("This is a complex tweet. It contains tweet and comment\nThe origin tweet(" + text_type + "):")
        print(origin_text)


def finalize_json_data():
    """
    Summarize all the data that collected to JSONs files
    :return:
    """
    print("Saving data...")
    # saves the tweets we didn't passed yet
    new_unlabeled_list = unlabeled[i:]
    create_json_dict_file(new_unlabeled_list, 'Temp files/' + TRANSLATED_JSON)
    create_json_dict_file(labeled, 'Temp files/labeled_tweets.json')
    create_json_dict_file(problematic_tweets, 'Temp files/problem_tweets.json')
    print("data saved!\n")


def labeling_report(total_signed_tweets, is_finish_labeling=False):
    """
    The report stage to the group
    :param total_signed_tweets: the number of the all tweets we passed and signed
    :param is_finish_labeling: the case of sending the mail in order to know the mail's subject
    :return:
    """
    mail_body = "{0} tweets total\n{1} labeled tweets and {2} problematic tweet. :-)".format(
        str(total_signed_tweets), str(len(labeled)), str(len(problematic_tweets)))
    print("Sending email to the teammates...\n")
    if not is_finish_labeling:
        send_report_by_email(mail_subject=NAME + "'s label progress", body_text=mail_body)
    else:
        send_report_by_email(mail_subject=NAME + "'s label progress - DONE!", body_text=mail_body)


def relative_subject_labeler():
    """
    The relative stage. we choose here what kind of the tweet
    :return: topic / person as string
    """
    # user input for the relativity label
    relativity_type = input("Is the tweet relative to a person or topic?\n"
                            "Please press 0 - for person, 1 - for topic\n")
    # checks if the input is legal
    if re.search('^[0-1]', relativity_type) and relativity_type.__len__() == 1:
        if relativity_type == '1':
            return "topic"
        else:
            return "person"
    # run the input function again until the input is legal
    else:
        print("you pressed wrong number, please try again")
        return relative_subject_labeler()


def tweet_pos_neg_labeler(number_of_labelers=0, previous_label=0):
    """
    The positive or negative labeling stage.
    we choose here the sign of the tweet from 1 up to 5
    :return: positivity label as string
    """
    neg_pos = input("What is the level of positivity/negativity?\n"
                    "Please press 1 (negative) up to 5 (positive)\n")
    # checks if the input is legal
    if re.search('^[1-5]', neg_pos) and neg_pos.__len__() == 1:
        if int(neg_pos) >= 1 or int(neg_pos) <= 5:
            prev_val = float(previous_label * number_of_labelers)
            return (prev_val + int(neg_pos)) / (number_of_labelers + 1)
    # run the input function again until the input is legal
    else:
        print("you typed wrong input, please try again")
        return tweet_pos_neg_labeler(number_of_labelers, previous_label)


def backup_files():
    """
    backup the file we writing
    :return:
    """
    separate_debug_print_small("Backup the files now")
    labeled_backup_dst = 'Temp files/backup/labeled_tweets_until_' + str(len(labeled)) + '.json'
    problem_backup_dst = 'Temp files/backup/problem_tweets_until_' + str(len(problematic_tweets)) + '.json'
    unlabeled_backup_dst = 'Temp files/backup/unlabeled_tweets_until_' + str(len(unlabeled) - i) + '.json'
    create_json_dict_file(labeled, labeled_backup_dst)
    create_json_dict_file(problematic_tweets, problem_backup_dst)
    create_json_dict_file(unlabeled[i:], unlabeled_backup_dst)
    separate_debug_print_small("Backup done")


def main_labeler(t):
    labeler_status = True
    separate_debug_print_small("starting tweet's labeler")
    print_tweet_data(t)
    while labeler_status:
        user_action = input("\nDo you want to label this tweet or skip to consult with the teammates?\n"
                            "Please press:\n0 - for collect to teammates\n"
                            "1 - for continue labeling\n2 - see the text again\n3 - skip this tweet\n")
        if user_action == '0':
            problematic_tweets.append(t)
            labeler_status = False
        elif user_action == '1':
            # cur_tweet_labeler(t) - Not in use
            # creating the label dictionary we want to append to the translated json
            t["label"] = {'positivity': tweet_pos_neg_labeler(),
                          'relative subject': relative_subject_labeler()}
            if 'quoted_status' in t:
                print("This tweet is a comment for the following tweet.\n"
                      "Please label the previous tweet relatively to the following tweet:")
                print_tweet_data(t['quoted_status'], quoted=True)
                t["label"].update({'origin_relative_positivity': tweet_pos_neg_labeler(),
                                   'origin_relative_relative subject': relative_subject_labeler()})
            labeled.append(t)
            labeler_status = False
        elif user_action == '2':
            print_tweet_data(t)
        elif user_action == '3':
            # doesn't let the user to skip this tweet in case JsonManger created it
            if 'JsonManager' in t:
                print("This is an important tweet. You can't skip it")
                continue
            print("Bye bye, you unuseful tweet, TFIEE!\n")
            labeler_status = False
        else:
            print("You entered wrong input!\nYou should enter 0, 1 or 2\nPlease try again\n")


if __name__ == '__main__':
    script_opener("Tweet Labeler")
    unlabeled, labeled, problematic_tweets = initialize_data()

    # added temporary JSON checker in order to modify the unlabeled data with quotes
    json_manager = JsonManager(unlabeled)
    unlabeled = json_manager.create_json_with_quotes()

    if len(unlabeled) == 0:
        print("You have empty json!\nPlease Check your tweet's file")

    i = 0
    # run main while loop as far as the is unlabeled tweets
    while i < len(unlabeled):
        # the backup staging every BACKUP_RATIO constant
        if i % BACKUP_RATIO == 0 and not i == 0:
            backup_files()

        user_main_choose = input("\nDo you want to label a tweet?\nPlease press:\n0 - for no\n1 - for yes\n")

        if user_main_choose == '1':
            separate_debug_print_big("label number " + str(i + 1))
            main_labeler(unlabeled[i])

        elif user_main_choose == '0':
            # sends report in order the user want
            if input("Do you want to share your progress?\n"
                     "Please press 1 - for yes, or anything else - for no\n") == '1':
                if NAME == "":
                    NAME = input("\nYou didn't entered your name. Please enter your name now:\n")
                labeling_report(total_signed_tweets=len(labeled) + len(problematic_tweets))
            finalize_json_data()
            print("Goodbye Chiquititas!!!")
            exit(0)

        else:
            print("You entered wrong input!\nPlease run again\n")
            # sub the i in order to run the same tweet again
            i -= 1

        i += 1

    # in case the translated file ended
    if i != 0:
        if NAME == "":
            NAME = input("\nYou didn't entered your name. Please enter your name now:\n")
        labeling_report(total_signed_tweets=len(labeled) + len(problematic_tweets), is_finish_labeling=True)
        finalize_json_data()
