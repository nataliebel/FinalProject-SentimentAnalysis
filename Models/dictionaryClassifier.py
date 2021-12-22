import re
from datetime import datetime

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

datetime_object = datetime.now()
dt = datetime_object.strftime("%d_%m_%H_%M")
en_he = 'hebrew'
tweets_file_name = 'C:/Users/yonat/PycharmProjects/SentimentAnalysisProject/Models/Data/eng_tweets_df.csv'


# NOTE- I CHANGED BULL**** TO BULL AND F**K TO REAL WORD

def get_pos_neg_tweets_df(fname, language='hebrew'):
    tweets_df = pd.read_csv(fname)
    tweets_df['tweet_words'] = create_tweet_words(tweets_df['tweet'], language)
    return tweets_df


# counts number of words from ls_b in a
def num_of_words_in_vec(a, ls_b):
    count = 0
    for it in a:
        if it in ls_b:
            count += 1
    return count


# removes symbols and any non words from tweets
def create_tweet_words(features):
    processed_features = []
    for sentence in range(0, len(features)):
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

        # remove all single characters
        processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

        # Remove single characters from the start
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

        # Removing prefixed 'b'
        processed_feature = re.sub(r'^b\s+', '', processed_feature)

        # Converting to Lowercase
        processed_feature = processed_feature.lower()

        ls_fet = processed_feature.split(' ')
        while "" in ls_fet:
            ls_fet.remove("")

        processed_features.append(ls_fet)

    return processed_features


def classify(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def is_correct(x, y):
    if x == 0:
        return 0
    if x == y:
        return 1
    else:
        return -1


def create_df_and_vocab_ls(p_wrds_fname, n_wrds_fname):
    # fPos = "twitter_samples/positive_tweets1.json"
    # fNeg = "twitter_samples/negative_tweets1.json"
    # neg_pos_tweets = positive_tweets + negative_tweets
    # random.shuffle(neg_pos_tweets)
    # neg_pos_tweets = pd.DataFrame(neg_pos_tweets)
    is_trans = False
    # positive_tweets, negative_tweets = mUtils.get_tweets(fPos, fNeg)
    tweets_fname = tweets_file_name
    neg_pos_tweets = get_pos_neg_tweets_df(tweets_fname)

    pos_wrds_fname = f'vocab_classifier/vocabularies/{p_wrds_fname}'
    neg_wrds_fname = f'vocab_classifier/vocabularies/{n_wrds_fname}'

    with open(pos_wrds_fname, encoding="utf8") as f:
        pos_words = f.read().split('\n')

    with open(neg_wrds_fname, encoding="utf8") as f1:
        neg_words = f1.read().split('\n')

    tweet_text = neg_pos_tweets.loc[:, 'tweet'].values
    ids = neg_pos_tweets.loc[:, 'id_str'].values
    labels = neg_pos_tweets.loc[:, 'label'].values
    tweet_words = create_tweet_words(tweet_text)
    tweet_wrds_df = pd.DataFrame(data={'id': ids, 'tweet_words': tweet_words, 'label': labels})
    return tweet_wrds_df, pos_words, neg_words


# creates lists of vocabs
def create_vocab_ls(pos_fname, neg_fname):
    with open(pos_fname, encoding="utf8") as f:
        pos_words = f.read().split('\n')

    with open(neg_fname, encoding="utf8") as f1:
        neg_words = f1.read().split('\n')
    return pos_words, neg_words


# classifies accoriding to word count
def classify_by_vocab(tweets_df, pos_words, neg_words):
    tweets_df['pos_words_count'] = tweets_df.apply(lambda x: num_of_words_in_vec(x['tweet_words'], pos_words),
                                                   axis=1)
    tweets_df['neg_words_count'] = tweets_df.apply(lambda x: num_of_words_in_vec(x['tweet_words'], neg_words),
                                                   axis=1)
    tweets_df['vocab_words_dif'] = tweets_df.pos_words_count - tweets_df.neg_words_count
    tweets_df['classifier'] = tweets_df.vocab_words_dif.apply(classify)
    # tweets_df['clas_correct'] = tweets_df.apply(lambda x: is_correct(x['classifier'], x['label']), axis=1)
    tweets_df['new_label'] = tweets_df['label']
    tweets_df['new_label'] = np.where(tweets_df['label'] > 3, 1, tweets_df['new_label'])
    tweets_df['new_label'] = np.where(tweets_df['label'] < 3, -1, tweets_df['new_label'])
    tweets_df['new_label'] = np.where(tweets_df['new_label'] == 3, 0, tweets_df['new_label'])
    tweets_df['clas_correct'] = (tweets_df.new_label == tweets_df.classifier).astype(int)
    tweets_df['clas_correct'].loc[tweets_df['clas_correct'] == 0] = -1
    # removed row because label 3 should have 0 difference
    # tweets_df['clas_correct'].loc[tweets_df['classifier'] == 0] = 0

    return tweets_df


# regular word count classification
def run_classification(fout_name, tweets_df_fname, pos_words_fname, neg_words_fname):
    pos_words, neg_words = create_vocab_ls(pos_words_fname, neg_words_fname)
    tweets_df = get_pos_neg_tweets_df(tweets_df_fname)
    res_df = classify_by_vocab(tweets_df, pos_words, neg_words)
    clas_series = res_df.pivot_table(index=['clas_correct'], aggfunc='size')
    clas_df = pd.DataFrame(data={'is_correct': clas_series.index.values, 'count': clas_series})
    clas_df.reset_index(drop=True, inplace=True)
    clas_df.loc[clas_df['is_correct'] == 1, 'is_correct'] = 'correct'
    clas_df.loc[clas_df['is_correct'] == -1, 'is_correct'] = 'incorrect'
    clas_df.loc[clas_df['is_correct'] == 0, 'is_correct'] = 'uncovered'
    clas_df.to_csv(f'{fout_name}_{dt}.csv')


# creates list of words from ls_b in a
def words_in_vec(a, ls_b):
    ls_a = a.split(' ')
    while "" in ls_a:
        ls_a.remove("")
    words_ls = []
    for it in ls_a:
        if it in ls_b:
            words_ls.append(it)
    return words_ls


# adds columns with vocab words
def add_vocab_columns(tweets_df, pos_words, neg_words):
    tweets_df['pos_words'] = tweets_df.apply(lambda x: words_in_vec(x['tweet_words'], pos_words),
                                             axis=1)
    tweets_df['neg_words'] = tweets_df.apply(lambda x: words_in_vec(x['tweet_words'], neg_words),
                                             axis=1)
    return tweets_df


# seemingly not used
def create_tweets_w_vocab_df(pos_words_fname, neg_words_fname):
    tweets_df, pos_words, neg_words = create_df_and_vocab_ls(pos_words_fname, neg_words_fname)
    # tweets_df = add_vocab_columns(tweets_df, pos_words, neg_words)
    pos_df, neg_df = has_vocab_words(tweets_df, pos_words, neg_words)
    return pos_df, neg_df, pos_words, neg_words


# seemingly not used
def has_vocab_words(df, pos_words=None, neg_words=None):
    pos_df = pd.DataFrame()
    neg_df = pd.DataFrame()

    for p in pos_words:
        df['w'] = p
        df['has_word'] = [c in l for c, l in zip(df['w'], df['tweet_words'])]
        a = df.loc[df.has_word == True].loc[df.label == 1]
        if a.empty:
            continue
        a.drop(columns=['has_word'], inplace=True)
        pos_df = pos_df.append(a)
    for n in neg_words:
        df['w'] = n
        df['has_word'] = [c in l for c, l in zip(df['w'], df['tweet_words'])]
        a = df.loc[df.has_word == True].loc[df.label == -1]
        if a.empty:
            continue
        a.drop(columns=['has_word'], inplace=True)
        neg_df = neg_df.append(a)
    pos_df.reset_index(drop=True, inplace=True)
    neg_df.reset_index(drop=True, inplace=True)
    return pos_df, neg_df


def print_words_df(df):
    pd.set_option('display.max_colwidth', -1)
    for column in df.columns:
        s = df.loc[df[column].notna()][column]
        print(column, ":", s)


# gets most frequent words form tweets acording to tfidf
def get_most_frequent(df, vocab1, vocab2):
    df['tweet_words'] = df['tweet_words'].astype(str)
    # creating tfidf vecotr
    stop_words = set(stopwords.words('english'))
    cv = CountVectorizer(max_df=0.85, stop_words=stop_words)
    word_count_vector = cv.fit_transform(df['tweet_words'])
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    words = cv.get_feature_names()
    # creating df with word count
    word_count = word_count_vector.toarray().sum(axis=0)
    word_count_df = pd.DataFrame(data={'word': words, 'word_count': word_count})
    # finding words that are already in vocab1
    word_count_df['to_remove'] = word_count_df['word'].isin(vocab1)
    word_count_df.drop(word_count_df.loc[word_count_df.to_remove == True].index, inplace=True)
    # finding words that are already in vocab2
    word_count_df['to_remove'] = word_count_df['word'].isin(vocab2)
    # removing words from either vocab
    word_count_df.drop(word_count_df.loc[word_count_df.to_remove == True].index, inplace=True)
    word_count_df.drop(columns=['to_remove'], inplace=True)
    word_count_df.sort_values(by='word_count', inplace=True, ascending=False)
    word_count_df.reset_index(inplace=True, drop=True)
    return word_count_df


# finds words to add to vocabulary
def make_special_words(pos_df, neg_df):
    pos_v = pos_df['word'].to_list()
    neg_v = neg_df['word'].to_list()
    pos_df['is_special'] = ~pos_df['word'].isin(neg_v)
    neg_df['is_special'] = ~neg_df['word'].isin(pos_v)
    # was better without this and just took new special words by hand
    # finding 'almost special words' i.e appear 3.5 times more in one than the other
    pos_neg_words = pos_df.loc[pos_df['word'].isin(neg_v)]
    pos_neg_words.rename(columns={'word_count': 'pos_word_count'}, inplace=True)
    pos_neg_words.drop(columns=['is_special'], inplace=True)
    pos_neg_words = pos_neg_words.merge(neg_df, on='word')
    pos_neg_words.rename(columns={'word_count': 'neg_word_count'}, inplace=True)
    pos_neg_words.drop(columns=['is_special'], inplace=True)
    pos_neg_words['pos_special'] = pos_neg_words['pos_word_count'] >= pos_neg_words['neg_word_count'] * 3.5
    pos_neg_words['neg_special'] = pos_neg_words['neg_word_count'] >= pos_neg_words['pos_word_count'] * 3.5
    # keeping only special words from original special words df
    pos_df = pos_df[pos_df['is_special'] == True]
    neg_df = neg_df[neg_df['is_special'] == True]
    new_pos_special = pos_neg_words.loc[pos_neg_words['pos_special'] == True]
    new_pos_special.drop(columns=['neg_word_count', 'neg_special'], inplace=True)
    new_pos_special.rename(columns={'pos_word_count': 'word_count', 'pos_special': 'is_special'}, inplace=True)
    pos_df = pos_df.append(new_pos_special)
    new_neg_special = pos_neg_words.loc[pos_neg_words['neg_special'] == True]
    new_neg_special.drop(columns=['pos_word_count', 'pos_special'], inplace=True)
    new_neg_special.rename(columns={'neg_word_count': 'word_count', 'neg_special': 'is_special'}, inplace=True)
    neg_df = neg_df.append(new_neg_special)
    pos_df = pos_df.loc[pos_df['word_count'] >= 5]
    neg_df = neg_df.loc[neg_df['word_count'] >= 5]
    return pos_df, neg_df


def get_bad_word(df, vocab):
    bad_words_ls = []
    for index, row in df.iterrows():
        words = row.tweet_words
        added_word = False
        for w in words:
            if w in vocab:
                bad_words_ls.append(w)
                added_word = True
                break
        if not added_word:
            bad_words_ls.append(None)
        # will only get here if it doesn't break
    df['bad_word'] = bad_words_ls
    return df


# removes words that classify incorrectly and saves the new vocabulary in the files specified in function
def remove_bad_words(tweets_df, pos_words, neg_words):
    res_df = classify_by_vocab(tweets_df, pos_words, neg_words)
    wrong_class = res_df.loc[res_df['clas_correct'] == -1]
    one_wrong_word_df = wrong_class.loc[wrong_class['label'] == 1].loc[wrong_class['neg_words_count'] == 1]
    one_wrong_word_df = one_wrong_word_df.append(
        wrong_class.loc[wrong_class['label'] == -1].loc[wrong_class['pos_words_count'] == 1])
    pos_one_wrong_word_df = one_wrong_word_df.loc[one_wrong_word_df['label'] == 1]
    neg_one_wrong_word_df = one_wrong_word_df.loc[one_wrong_word_df['label'] == -1]
    pos_one_wrong_word_df = get_bad_word(pos_one_wrong_word_df, neg_words)
    neg_one_wrong_word_df = get_bad_word(neg_one_wrong_word_df, pos_words)
    check_df = res_df.loc[res_df['vocab_words_dif'] == 1]
    pos_check_df = check_df.loc[check_df['label'] == 1].loc[check_df['classifier'] == 1]
    neg_check_df = check_df.loc[check_df['label'] == -1].loc[check_df['classifier'] == -1]
    # the bad words are in opposite lists, that's why they are bad
    neg_bad_words = pos_one_wrong_word_df['bad_word'].to_list()
    pos_bad_words = neg_one_wrong_word_df['bad_word'].to_list()
    # checking whether there are 'bad' word that are crucial for other classifications,
    # i.e without them a tweet will be uncovered
    pos_check_df = get_bad_word(pos_check_df, pos_bad_words)
    neg_check_df = get_bad_word(neg_check_df, neg_bad_words)
    pos_check_list = pos_check_df['bad_word'].to_list()
    neg_check_list = neg_check_df['bad_word'].to_list()
    for w in pos_bad_words:
        if w in pos_check_list:
            pos_bad_words.remove(w)
    for w in neg_bad_words:
        if w in neg_check_list:
            neg_bad_words.remove(w)
    new_pos_fname = f'vocab_classifier/vocabularies/{en_he}/positive_words_clean_{en_he}_{dt}.txt'
    new_neg_fname = f'vocab_classifier/vocabularies/{en_he}/negative_words_clean_{en_he}_{dt}.txt'
    remove_words_from_file(pos_bad_words, pos_words, new_pos_fname)
    remove_words_from_file(neg_bad_words, neg_words, new_neg_fname)
    return new_pos_fname, new_neg_fname


# removes bad words and rewrites final vocav to file
def remove_words_from_file(words_ls, vocab, fname):
    for w in words_ls:
        if w in vocab:
            vocab.remove(w)
    with open(fname, 'w+', encoding='utf-8') as filehandle:
        for word in vocab:
            filehandle.write('%s\n' % word)


def calc_pearson_corrolation(df):
    words = df.columns[3:]
    corrs = []
    for w in words:
        corr = df.loc[:, w].corr(df.loc[:, 'label'])
        corrs.append(corr)
    corrs_res = pd.DataFrame(data={'word': words, 'p_cor': corrs})
    corrs_res['p_cor'] = np.where(corrs_res['p_cor'].isna(), 0, corrs_res['p_cor'])
    corrs_res['p_cor'] = np.where(corrs_res['p_cor'] < 0, 0, corrs_res['p_cor'])
    return corrs_res


# creates df with word columns if tweet contains word
def create_tweet_vocab_df(tweets_data, vocab):
    ids = tweets_data.loc[:, 'id_str']
    tweets = tweets_data.loc[:, 'tweet_words']
    labels = tweets_data.loc[:, 'label']
    tweet_vocab_df = pd.DataFrame(data={'id': ids, 'tweet_words': tweets, 'label': labels})
    tweet_vocab_df['tweet_words'] = ['|'.join(map(str, l)) for l in tweet_vocab_df['tweet_words']]
    # checking appearance of each word in every tweet
    for w in vocab:
        try:
            tweet_vocab_df[w] = tweet_vocab_df['tweet_words'].str.contains(w).astype(int)
        except:
            b = 1
    return tweet_vocab_df


def avg_weights(tweet, word_weights_df):
    word_weights_df['is_in_tweet'] = word_weights_df['word'].isin(tweet)
    avg = word_weights_df['p_cor'].loc[word_weights_df['is_in_tweet'] == True].mean()
    return avg


def sum_weights(tweet, word_weights_df):
    if type(tweet) is float:
        return
    word_weights_df['is_in_tweet'] = word_weights_df['word'].isin(tweet.split(' '))
    sum = word_weights_df['p_cor'].loc[word_weights_df['is_in_tweet'] == True].sum()
    return sum


def classify_by_weights(tweets_df, pos_words_weights, neg_words_weights):
    # finding average if positive and negative weights
    # tweets_df['pos_words_avg'] = tweets_df.apply(lambda x: avg_weights(x['tweet_words'], pos_words_weights), axis=1)
    # tweets_df['neg_words_avg'] = tweets_df.apply(lambda x: avg_weights(x['tweet_words'], neg_words_weights), axis=1)
    # tweets_df['pos_words_avg'] = np.where(tweets_df['pos_words_avg'].isna(), 0, tweets_df['pos_words_avg'])
    # tweets_df['neg_words_avg'] = np.where(tweets_df['neg_words_avg'].isna(), 0, tweets_df['neg_words_avg'])
    # tweets_df['vocab_words_dif'] = tweets_df.pos_words_avg - tweets_df.neg_words_avg
    tweets_df['pos_words_sum'] = tweets_df.apply(lambda x: sum_weights(x['tweet_words'], pos_words_weights), axis=1)
    tweets_df['neg_words_sum'] = tweets_df.apply(lambda x: sum_weights(x['tweet_words'], neg_words_weights), axis=1)
    # changing nan values to 0
    tweets_df['pos_words_sum'] = np.where(tweets_df['pos_words_sum'].isna(), 0, tweets_df['pos_words_sum'])
    tweets_df['neg_words_sum'] = np.where(tweets_df['neg_words_sum'].isna(), 0, tweets_df['neg_words_sum'])
    tweets_df['vocab_words_dif'] = tweets_df.pos_words_sum - tweets_df.neg_words_sum
    tweets_df['classifier'] = tweets_df.vocab_words_dif.apply(classify)
    # tweets_df['clas_correct'] = tweets_df.apply(lambda x: is_correct(x['classifier'], x['label']), axis=1)
    tweets_df['clas_correct'] = (tweets_df.new_label == tweets_df.classifier).astype(int)
    tweets_df['clas_correct'].loc[tweets_df['clas_correct'] == 0] = -1
    # tweets_df['clas_correct'].loc[tweets_df['classifier'] == 0] = 0

    return tweets_df


# classifies according to weights (like pearson cor)
def run_weighted_classification(tweets_df_fname, pos_weights_df_fname, neg_weights_df_fname):
    pos_words_weights = pd.read_csv(pos_weights_df_fname)
    neg_words_weights = pd.read_csv(neg_weights_df_fname)
    tweets_df = get_pos_neg_tweets_df(tweets_df_fname)
    tweets_df['new_label'] = tweets_df['label']
    tweets_df['new_label'] = np.where(tweets_df['label'] > 3, 1, tweets_df['new_label'])
    tweets_df['new_label'] = np.where(tweets_df['label'] < 3, -1, tweets_df['new_label'])
    tweets_df['new_label'] = np.where(tweets_df['label'] == 3, 0, tweets_df['new_label'])
    res_df = classify_by_weights(tweets_df, pos_words_weights, neg_words_weights)
    return res_df


def get_tweets_weights_feature(tweets_df, language):
    pos_weights_df_fname = f"C:\SentimentAnalysisProject\\vocabularies\\final_extended_neg_pearson_{language}.csv"
    neg_weights_df_fname = f"C:\SentimentAnalysisProject\\vocabularies\\final_extended_pos_pearson_{language}.csv"
    # pos_weights_df_fname = r'C:\Users\dembo\Documents\Computer Science\Third Year\Project\Sentiment Analysis Project\vocabularies\final_extended_pos_pearson_heb.csv'
    # neg_weights_df_fname = r'C:\Users\dembo\Documents\Computer Science\Third Year\Project\Sentiment Analysis Project\vocabularies\final_extended_neg_pearson_heb.csv'
    pos_words_weights = pd.read_csv(pos_weights_df_fname)
    neg_words_weights = pd.read_csv(neg_weights_df_fname)
    tweets_df['new_label'] = tweets_df['label']
    tweets_df['new_label'] = np.where(tweets_df['label'] > 3, 1, tweets_df['new_label'])
    tweets_df['new_label'] = np.where(tweets_df['label'] < 3, -1, tweets_df['new_label'])
    tweets_df['new_label'] = np.where(tweets_df['label'] == 3, 0, tweets_df['new_label'])
    tweets_df['pos_words_sum'] = tweets_df.apply(lambda x: sum_weights(x['tweet_words'], pos_words_weights), axis=1)
    tweets_df['neg_words_sum'] = tweets_df.apply(lambda x: sum_weights(x['tweet_words'], neg_words_weights), axis=1)
    # changing nan values to 0
    tweets_df['pos_words_sum'] = np.where(tweets_df['pos_words_sum'].isna(), 0, tweets_df['pos_words_sum'])
    tweets_df['neg_words_sum'] = np.where(tweets_df['neg_words_sum'].isna(), 0, tweets_df['neg_words_sum'])
    tweets_df['vocab_feature'] = tweets_df.pos_words_sum - tweets_df.neg_words_sum
    return tweets_df


# runs and saves weighted (i.e pearson cor+ext) classification
def run_and_save_weighted_clas(fout_name, tweets_df_fname, pos_weights_df_fname, neg_weights_df_fname):
    res_df = run_weighted_classification(tweets_df_fname, pos_weights_df_fname, neg_weights_df_fname)
    clas_series = res_df.pivot_table(index=['clas_correct'], aggfunc='size')
    clas_df = pd.DataFrame(data={'is_correct': clas_series.index.values, 'count': clas_series})
    clas_df.reset_index(drop=True, inplace=True)
    clas_df.loc[clas_df['is_correct'] == 1, 'is_correct'] = 'correct'
    clas_df.loc[clas_df['is_correct'] == -1, 'is_correct'] = 'incorrect'
    clas_df.loc[clas_df['is_correct'] == 0, 'is_correct'] = 'uncovered'
    clas_df.to_csv(f'vocab_classifier/results/real_data/{fout_name}_{en_he}_{dt}.csv')


def create_pearson_cor_files(tweets_fname, pos_voab_fname, neg_vocab_fname):
    pos_vocab, neg_vocab = create_vocab_ls(pos_voab_fname, neg_vocab_fname)
    tweets_df = get_pos_neg_tweets_df(tweets_fname)
    ln = len(tweets_df.index)

    t_size = int(0.8 * ln)
    train = tweets_df.iloc[:ln]
    test = tweets_df.iloc[t_size:]

    # for running on full list
    # train = tweets_df
    # test = tweets_df
    tweets_neg_vocab_df = create_tweet_vocab_df(train, neg_vocab)
    tweets_pos_vocab_df = create_tweet_vocab_df(train, pos_vocab)
    neg_pearson_cor = calc_pearson_corrolation(tweets_neg_vocab_df)
    # removing last row - empty
    neg_pearson_cor.drop(neg_pearson_cor.tail(1).index, inplace=True)
    pos_pearson_cor = calc_pearson_corrolation(tweets_pos_vocab_df)
    # removing last row - empty
    pos_pearson_cor.drop(pos_pearson_cor.tail(1).index, inplace=True)
    pos_pearson_cor.to_csv(f'vocab_classifier/results/real_data/pos_pearson_cor_{en_he}_{dt}.csv')
    neg_pearson_cor.to_csv(f'vocab_classifier/results/real_data/neg_pearson_cor_{en_he}_{dt}.csv')

    return pos_pearson_cor, neg_pearson_cor


# create new dictionary with new frequent words
def clean_and_basic_update_vocab(orig_pos_vocab, orig_neg_vocab):
    tweets_df, pos_words, neg_words = create_df_and_vocab_ls(orig_pos_vocab, orig_neg_vocab)
    new_pos_fname, new_neg_fname = remove_bad_words(tweets_df, pos_words, neg_words)
    pos_words, neg_words = create_vocab_ls(new_pos_fname, new_neg_fname)
    pos_words_df = tweets_df.loc[tweets_df['new_label'] == 1]
    neg_words_df = tweets_df.loc[tweets_df['new_label'] == -1]
    new_pos_words = get_most_frequent(pos_words_df, pos_words, neg_words)
    new_neg_words = get_most_frequent(neg_words_df, neg_words, pos_words)
    new_pos_words, new_neg_words = make_special_words(new_pos_words, new_neg_words)
    new_pos_words.to_csv(f'vocab_classifier/vocabularies/{en_he}/new_pos_words_{dt}.csv')
    new_neg_words.to_csv(f'vocab_classifier/vocabularies/{en_he}/new_neg_words_{dt}.csv')


# runs pearson corr weights classification on test
def run_weighted_clas_on_test(test_fname, pos_p_cor_fname, neg_p_cor_fname):
    test = pd.read_csv(f'vocab_classifier/{test_fname}.csv').drop(columns=['Unnamed: 0'])
    test.replace({'\'': '', '\[': '', '\]': ''}, regex=True, inplace=True)
    test['tweet_words'] = test['tweet_words'].str.split(', ')
    pos_cors = pd.read_csv(f'vocab_classifier/results/{pos_p_cor_fname}.csv')
    neg_cors = pd.read_csv(f'vocab_classifier/results/{neg_p_cor_fname}.csv')
    # run_weighted_classification("weighted_pearson_vocab_test", test, pos_cors, neg_cors)


if __name__ == "__main__":
    tweets_df = get_pos_neg_tweets_df(tweets_file_name)
    get_tweets_weights_feature(tweets_df, 'english')

a = 1
