import json

from tweepy.streaming import StreamListener
from shutil import copyfile
from support.Utils import send_report_by_email, separate_debug_print_small

EMAIL_MAIL_MSG = "The number of the tweets we have been accumulating so far is: "

# constants
SAVING_CONSTANT = 100
EMAIL_CONSTANT = 1000


# # # # TWITTER STREAM LISTENER # # # #


class TwitterListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """

    def __init__(self, fetched_tweets_filename, tweets_number):
        self.fetched_tweets_filename = fetched_tweets_filename
        self.c = tweets_number

    def on_data(self, data):
        try:
            all_data = json.loads(data)
            self.append_json(all_data)
            return True
        except BaseException as e:

            send_report_by_email(mail_subject="Lite Error Accrued!!!", body_text='Lite error, not big deal\nFYI')
            print("Error on_data %s" % str(e))
        return True

    def report_and_modify(self):
        """
        this method for reporting and tracking the result.
        the results it send via email
        :return:
        """
        self.c = self.c + 1
        # prints the tweets was download bu now
        print("Tweet number " + str(self.c) + " was written into the JSON.")
        # save new file every 'SAVING_CONSTANT' as new file in the archive
        if self.c % SAVING_CONSTANT == 0:
            separate_debug_print_small("Saving JSON by now")
            dst = 'archive/tweets_' + str(self.c) + '.json'
            copyfile('tweets.json', dst)
        # sends email with the tweets by now every 'EMAIL_CONSTANT'
        if self.c % EMAIL_CONSTANT == 0:
            separate_debug_print_small("sending email...")
            send_report_by_email(mail_subject="Tweets Download Status", body_text=EMAIL_MAIL_MSG+str(self.c))
            separate_debug_print_small("summary until now:")
            print("number of tweets: " + str(self.c))
            separate_debug_print_small("end of summary")

    def write_json(self, data):
        with open(self.fetched_tweets_filename, 'w', encoding="utf-8") as tf:
            json.dump(data, tf, ensure_ascii=False, indent=3)
        self.report_and_modify()


    def append_json(self, data):
        with open(self.fetched_tweets_filename, 'r', encoding="utf-8") as json_file:
            old_json = json.load(json_file)
            temp = old_json['tweets']
            temp.append(data)
            self.write_json(old_json)

    def on_error(self, status):
        if status == 420:
            # Returning False on_data method in case rate limit occurs.

            send_report_by_email(mail_subject="Error Accrued!!!", body_text='ERROR,\nCall Gidi now!!!')

            return False
        else:
            send_report_by_email(mail_subject="Error Accrued!!!\n error number " + str(status) + ".", body_text='ERROR,\nCall Gidi now!!!')
        print(status)
