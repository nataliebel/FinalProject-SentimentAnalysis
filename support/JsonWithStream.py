"""
This script takes some big json and creates small one with the needed fields
Here, the needed fields are: id, id_str, text, full_text, label
"""
import ijson

from support.Utils import create_json_dict_file

# the file we want to open with stream
JFILE = r"C:\Users\dembo\Documents\Computer Science\Third Year\Project\Sentiment Analysis Project\Models\Data\No Labeled Translated Tweets 2.json"

dataset = ijson.parse(open(JFILE, encoding="utf8"))

tweet_dict = {}
text_dict = {}

k = None
sub_k = None
v = None
sub_v = None

is_reading_tweet = False
start_list = False
new_data_set = list()

for prefix, type_of_object, value in dataset:
    # checks if we need to create new tweet in the new list or not
    if prefix == 'tweets.item' and type_of_object == 'start_map':
        is_reading_tweet = True
    elif prefix == 'tweets.item' and type_of_object == 'end_map':
        is_reading_tweet = False
        new_data_set.append(tweet_dict)
        tweet_dict = {}

    if is_reading_tweet:
        """
        Those conditions checks the values in order to know
        what fields and what values to create in the new tweet
        """
        # id
        if prefix == 'tweets.item' and type_of_object == 'map_key' and value == 'id':
            k = value
        elif prefix == 'tweets.item.id' and type_of_object == 'number':
            v = value
            tweet_dict[k] = v

        # id_str
        elif prefix == 'tweets.item' and type_of_object == 'map_key' and value == 'id_str':
            k = value
        elif prefix == 'tweets.item.id_str' and type_of_object == 'string':
            v = value
            tweet_dict[k] = v

        # text
        elif prefix == 'tweets.item' and type_of_object == 'map_key' and value == 'text':
            text_dict = {}
            k = value
        elif prefix == 'tweets.item.text.item' and type_of_object == 'map_key':
            sub_k = value
        elif prefix == 'tweets.item.text.item.translatedText' and type_of_object == 'string':
            sub_v = value
            text_dict[sub_k] = sub_v
        elif prefix == 'tweets.item.text.item.input' and type_of_object == 'string':
            sub_v = value
            text_dict[sub_k] = sub_v
        elif prefix == 'tweets.item.text' and type_of_object == 'end_array':
            v = list()
            v.append(text_dict)
            tweet_dict[k] = v

        # full_text
        elif prefix == 'tweets.item' and type_of_object == 'map_key' and value == 'extended_tweet':
            text_dict = {}
            k = value
        elif prefix == 'tweets.item.extended_tweet.full_text.item' and type_of_object == 'map_key':
            sub_k = value
        elif prefix == 'tweets.item.extended_tweet.full_text.item.translatedText' and type_of_object == 'string':
            sub_v = value
            text_dict[sub_k] = sub_v
        elif prefix == 'tweets.item.extended_tweet.full_text.item.input' and type_of_object == 'string':
            sub_v = value
            text_dict[sub_k] = sub_v
        elif prefix == 'tweets.item.extended_tweet.full_text' and type_of_object == 'end_array':
            v = list()
            v.append(text_dict)
            tweet_dict[k] = {"full_text": v}

        # label
        elif prefix == 'tweets.item' and type_of_object == 'map_key' and value == 'label':
            text_dict = {}
            k = value
        elif prefix == 'tweets.item.label' and type_of_object == 'map_key':
            sub_k = value
        elif prefix == 'tweets.item.label.positivity' and type_of_object == 'string':
            sub_v = value
            text_dict[sub_k] = sub_v
        elif prefix == 'tweets.item.label.relative subject' and type_of_object == 'string':
            if value == "0":
                value = "person"
            elif value == "1":
                value = "topic"
            sub_v = value
            text_dict[sub_k] = sub_v
        elif prefix == 'tweets.item.label' and type_of_object == 'end_map':
            tweet_dict[k] = text_dict

# creates the new data set after the stream
create_json_dict_file(new_data_set, "No Labeled Translated Tweets 22.json")
