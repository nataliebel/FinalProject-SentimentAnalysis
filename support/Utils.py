import os
import smtplib
import json

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

ADMIN_SERVER = 'sentimentanalysispro@gmail.com'
ADMIN_PASSWORD = 'godnb52020!'

# email constants
MAIL_SUBJECT = 'Twitter json Report'

REPORT_NAME = 'Twitter_api/tweets.json'

# The project collaborators and their emails
PROJECT_GROUP = {
    'Gidi': "gidemsky26@gmail.com",
    'Daniella The Sis': "Daniella.kirshenbaum@gmail.com",
    'Oriya Aharon The best': "oriya717@gmail.com",
    'Batel Cohen the princess': "batel.cohen100@gmail.com",
    'Natali the mommy care': "Balulunatalie@gmail.com"
}


def script_opener(script_title):
    # this is a script console opener.
    print("################################################ Welcome ##################################################")
    print("#")
    print("#")
    print("#")
    print("#")
    print("#")
    print("############################################# " + script_title +
          " ###############################################\n")


def separate_debug_print_big(title):
    # long line separator for debug line console print
    print('\n-------------------------------------------' + title + '-------------------------------------------')


def separate_debug_print_small(title):
    # short line separator for debug line console print
    print('---------------' + title + '---------------')


def get_group_emails(key=None):
    """
    Gets the asked member e-mail
    :param key: member name or list of names
    :return: the email of the person or of all the group
    """
    if key is None:
        return PROJECT_GROUP.values()
    try:
        # return the list of key's emails
        return [PROJECT_GROUP[i] for i in key]
    except KeyError:
        print("You entered wrong key name for email")
        print("return Gidi's email instead of the name you entered")
        return PROJECT_GROUP['Gidi']


def get_json_tweet_list(src_json_file):
    """
    Converts json file with the following structure: (Dict) tweets : list
    :param src_json_file: json file with the correct structure
    :return: list of the tweets
    """
    if type(src_json_file) is list:
        return src_json_file
    # in case there is no json file yet - open with empty list
    if not os.path.isfile(src_json_file):
        if not src_json_file.endswith('.json'):
            src_json_file = src_json_file + '.json'
        create_json_dict_file([], src_json_file)
    try:
        with open(src_json_file, 'r', encoding="utf-8") as json_file:
            json_all_dict_data = json.load(json_file)
        return json_all_dict_data['tweets']
    except IOError:
        print("File not accessible - please create or place the file")


def create_json_dict_file(json_list, json_file_name):
    """
    Convert the list to correct json file: (Dict) tweets : list
    :param json_list: list od tweets
    :param json_file_name: the destination file name
    :return: json file
    """
    json_all_list = {'tweets': json_list}
    if not json_file_name.endswith('.json'):
        json_file_name = json_file_name + '.json'
    with open(json_file_name, 'w', encoding='UTF-8') as fp:
        json.dump(json_all_list, fp, ensure_ascii=False, indent=3)
    print("The file " + str(json_file_name) + " has been created with "
          + str(check_tweets_number(json_list)) + " tweets.")


def check_tweets_number(src_json_file):
    """
    checks the number of the tweets in the json file
    :param src_json_file:
    :return: number of tweets
    """
    return len(get_json_tweet_list(src_json_file))


def send_report_by_email(mail_subject="No subject", body_text=None, file_path=None):
    """
    The sending e-mail function using gmail account
    :param mail_subject: The subject of the email to be written
    :param body_text: the body of the email (if needed)
    :param file_path: in case we have file - sends the file path to be sent
    :return: -
    """

    msg = MIMEMultipart()
    msg['From'] = ADMIN_SERVER
    msg['Subject'] = mail_subject

    msg.attach(MIMEText('Hey,\nThis is an automated e-mail report system.\n' + body_text, 'plain'))

    # in case there is file to be sent - prepare the file to email format
    if file_path is not None:
        attachment = open(file_path, 'rb')
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= " + file_path)
        msg.attach(part)

    # converts all the text and files to email format
    cur_mail = msg.as_string()

    # SMTP server configuration and settings
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()

    # log-in to the gmail account and sends the mail
    server.login(ADMIN_SERVER, ADMIN_PASSWORD)
    server.sendmail(ADMIN_SERVER, get_group_emails(), cur_mail)

    # close user and SMTP connection
    server.quit()


# convert list of json to list - temporary function
def temp_convert_json_to_list(json_name):
    tweets = []
    for line in open(json_name):
        tweet = json.loads(line)
        tweet["label"] = 1
        tweets.append(tweet)
    return tweets


def dir_checker_creator(path):
    """
    check a directory existence and create it in case it doesn't
    :param path: the path to create
    :return:
    """
    # create the full path from the place the src file is placed
    path = "./" + path
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            print("couldn't create %s folder\nPlease check it or create it manually\nExit the program for now!" % path)
            exit(1)
        else:
            print("Successfully created the directory %s \n" % path)


def json_files_collector(path):
    """
    accumulates all labeled files stored in LABELED_JSONS
    :return: list of all the file names
    """
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.json' in file:
                files.append(os.path.join(r, file))
    print('the number of files is ' + str(files.__len__()))
    return files


def marge_all_json_file(file_list):
    """
    for every file extend all the tweets it has
    :return: big all the tweets one after the other
    """
    list_content = list()
    for f in file_list:
        list_content.extend(get_json_tweet_list(f))
    print("the total number of tweets in this the final list is: " + str(list_content.__len__()))
    return list_content


def check_duplicate_tweet(json_with_duplicate):
    """
    This function looks for the same tweets in some json list
    If it finds, the function deletes it
    :param json_with_duplicate:
    :return: new tweets list without duplicate tweets
    """
    new_fixed_list = json_with_duplicate
    deleted = 0

    for tweet, i in zip(json_with_duplicate, range(json_with_duplicate.__len__())):
        id_to_check = tweet['id']
        j = 0

        for t in json_with_duplicate[i+1:]:
            if t['id'] == id_to_check:
                del new_fixed_list[i+1+j]
                deleted += 1
                j -= 1
            j += 1

    print("Number of deleted double labeled tweets: " + str(deleted))
    return new_fixed_list


def retweet_checker(json_list_to_check):
    """
    This function runs all over the source list and deletes all the retweets
    The deleted tweets will be saved as new file
    :param json_list_to_check: the source list t check
    :return: new list without retweets at all
    """
    print("\nPlease wait, checking if there are retweets to delete...")
    deleted_tweets = 0
    deleted_list_tweets = list()
    new_tweets_list = list()
    for t in json_list_to_check:
        if 'retweeted_status' not in t:
            new_tweets_list.append(t)
        else:
            deleted_list_tweets.append(t)
            deleted_tweets += 1
    print("Retweet status checked!")
    if deleted_tweets > 0:
        create_json_dict_file(deleted_list_tweets, 'Temp files/temp_deleted_retweets.json')
        print("{0} retweeted tweets has been removed from your tweet list\n".format(str(deleted_tweets)))
    else:
        print("No tweets has been removed. The JSON list is OK!\n")
    return new_tweets_list


def create_sub_json(src_json, destination_json_size, destination_file_number):
    """

    :param src_json:
    :param destination_json_size:
    :param destination_file_number:
    :return:
    """
    file_number = 0
    destination_list = list()
    src_json = get_json_tweet_list(src_json)
    max_tweet = destination_file_number * destination_json_size
    for t, i in zip(src_json[:max_tweet], range(src_json.__len__())):
        destination_list.append(t)
        if (i+1) % destination_json_size == 0:
            create_json_dict_file(json_list=destination_list, json_file_name=str(file_number)+'-tweets to label')
            file_number += 1
            destination_list.clear()


def create_sub_json_by_label(src_json, label_to_save):
    """
    in case we want to create another json acourding to the labels
    :param src_json:
    :param label_to_save:
    :return:
    """
    destination_list = list()
    src_json = get_json_tweet_list(src_json)
    for t in src_json:
        if int(t['label']['positivity']) == label_to_save:
            destination_list.append(t)
    create_json_dict_file(json_list=destination_list, json_file_name='label3-to label to 4 or 5')


def check_json_format(json_list):
    """
    This function checks the json fields and the values
    Changes the values to the correct values we need to use
    :param json_list:
    :return: the new correct list with the correct fields and values
    """
    checked_json_list = list()
    temp_text = list()
    for t in json_list:
        if not isinstance(t['text'], list):
            temp_text.append({'translatedText': None, 'input': t['text']})
            t['text'] = temp_text.copy()
            temp_text.clear()

        if 'extended_tweet' in t:
            if not isinstance(t['extended_tweet']['full_text'], list):
                temp_text.append({'translatedText': None, 'input': t['extended_tweet']['full_text']})
                t['extended_tweet']['full_text'] = temp_text.copy()
                temp_text.clear()

        if not isinstance((t['label']), list):
            t['label']['positivity'] = str(int(t['label']['positivity']))
        if t['label']["relative subject"] == 'subject':

            t['label']["relative subject"] = 'person'
        checked_json_list.append(t)

    print("The labeled json format checked")
    return checked_json_list
