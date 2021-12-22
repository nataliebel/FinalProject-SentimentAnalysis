import os
import sys

sys.path.append(r'C:\Users\User\PycharmProjects\SentimentAnalysisProject')

from tweepy import Stream

from Twitter_Api.FileExtracor import FileExtracor
from Twitter_Api.TwitterAuthenticator import TwitterAuthenticator
from Twitter_Api.TwitterListener import TwitterListener
from support.Utils import check_tweets_number, script_opener


# # # # TWITTER STREAMER # # # #
class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """

    def __init__(self):
        self.twitter_autenticator = TwitterAuthenticator()

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        tweets_number = 0
        # checks the correct tweets by now if the file exits
        if os.path.isfile('tweets.json'):
            tweets_number = check_tweets_number('tweets.json')
        # This handles Twitter authentication and the connection to Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename, tweets_number)
        auth = self.twitter_autenticator.authenticate_twitter_app()
        stream = Stream(auth, listener, tweet_mode='extended', include_retweets=True)
        # This line filter Twitter Streams to capture data by the keywords:
        stream.filter(track=hash_tag_list)


if __name__ == '__main__':
    # Authenticate using config.py and connect to Twitter Streaming API.

    script_opener("Tweets Downloading")
    file_extractor = FileExtracor()
    hash_tag_list = file_extractor.extract_file("key_words.txt")
    fetched_tweets_filename = "tweets.json"

    twitter_streamer = TwitterStreamer()
    twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)
