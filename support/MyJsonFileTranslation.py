from GoogleTranslate.GoogleTranslateAPI import GoogleTranslateAPI
from support.Utils import separate_debug_print_small, send_report_by_email, create_json_dict_file, get_json_tweet_list

LOGGING = True

MAIN_JSON_FILE = 'test-set.json'

# creates the google access API
google = GoogleTranslateAPI()

TARGET_LANG = 'en'
SRC_LANG = 'iw'


def json_translation(json_list):
    """
    Json's fields translation function.
    This function translates the following fields (if there are exist):
    + "text"
    + "user" -> "description"
    + "retweeted_status" -> "text"
    + "retweeted_status" -> "extended_tweet" -> "full_text"
    + "extended_tweet" -> "full_text"
    :param json_list: list we want to translate
    :return:
    """
    i = 0
    try:
        for tweet in enumerate(json_list):
            # in case we want to print the results
            i += 1
            if type(tweet[1]['text']) is list:
                if 'translatedText' in tweet[1]['text'][0]:
                    if tweet[1]['text'][0]['translatedText'] is not None:
                        print(str(i) + " -> Not to translate")
                        continue
                    else:
                        tweet[1]['text'] = tweet[1]['text'][0]['input']
                        if 'extended_tweet' in tweet[1]:
                            tweet[1]['extended_tweet']['full_text'] = tweet[1]['extended_tweet']['full_text'][0]['input']
            if LOGGING:
                separate_debug_print_small('The ' + str(i) + ' translation')
            # "text"
            tweet[1]['text'] = google.translate(tweet[1]['text'], TARGET_LANG, text_language=SRC_LANG, level=2)
            # if tweet[1]['user']['description'] is not None:
            #     # "user" -> "description"
            #     tweet[1]['user']['description'] = google.translate(tweet[1]['user']['description'], 'en',
            #                                                        text_language='iw', level=2)
            # if 'retweeted_status' in tweet[1]:
            #     # "retweeted_status" -> "text"
            #     tweet[1]['retweeted_status']['text'] = google.translate(tweet[1]['retweeted_status']['text'], 'en',
            #                                                             text_language='iw', level=2)
            #     if 'extended_tweet' in tweet[1]['retweeted_status']:
            #         # "retweeted_status" -> "extended_tweet" -> "full_text"
            #         tweet[1]['retweeted_status']['extended_tweet']['full_text'] = google.translate(
            #             tweet[1]['retweeted_status']['extended_tweet']['full_text'], 'en', text_language='iw')
            if 'extended_tweet' in tweet[1]:
                # "extended_tweet" -> "full_text"
                tweet[1]['extended_tweet']['full_text'] = google.translate(tweet[1]['extended_tweet']['full_text'],
                                                                           TARGET_LANG,
                                                                           text_language=SRC_LANG, level=2)
            if i % 2000 == 0:
                send_report_by_email(mail_subject="Translated Tweets",
                                     body_text=str(i) + ' tweets has successfully translated!')
        send_report_by_email(mail_subject="Translated Tweets",
                             body_text=str(i) + ' tweets has successfully translated!')

    except Exception:
        translated_tweets = json_list[:i]
        print(str(i))
        print("the Exceptions is: " + Exception)
        print(Exception.__name__)
        create_json_dict_file(translated_tweets, "translated exception tweets.json")


if __name__ == '__main__':
    #list_to_translate = get_json_tweet_list('Temp files/unlabeled JSON/' + MAIN_JSON_FILE)
    list_to_translate = get_json_tweet_list("test json for bootstraper.json")

    json_translation(list_to_translate)

    result_json_name = 'translated_' + MAIN_JSON_FILE

    create_json_dict_file(list_to_translate, result_json_name)
