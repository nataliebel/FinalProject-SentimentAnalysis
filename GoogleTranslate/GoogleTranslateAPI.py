"""
The main google API translation Class.
Class abilities:
    It can translate string and list to any supported language
    Print all the supported languages
    Detects the text language

    The main our using language are:
    Hebrew is 'iw'
    English is 'en'
    Arabic is 'ar'
"""
import os
import six

from support import Utils as Tool
from google.cloud import translate_v2 as translate
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\dembo\Documents\Computer Science\Third Year\Project\Sentiment Analysis Project\GoogleTranslate\keys\GoogleTranslateAPI-key.json"


class GoogleTranslateAPI(object):

    def __init__(self):
        self.translate_client = translate.Client()

    def translate(self, source_text, to_language, level=0, debug_print=False, model=None, text_language=None):
        """
        Translates text into the target language.
        :param level: The final return result level:
            0 - for only the translation text
            1 - as dict with translatedText
            2 - as dict with translatedText and the origin text
            4 - the full dict
        :param text_language: is the origin language of the text
        :param source_text: list or strings of strings to translate.
        :param to_language: The language to translate results into.
        :param debug_print: If we want to print the result
        :param model: 'base' or 'nmt'. By default it using NMT.
        :return: list. The translated to the destination language
        """

        translated_lines = list()

        # Modifies the text to utf-8 in case it doesn't
        if isinstance(source_text, six.binary_type):
            source_text = source_text.decode('utf-8')

        # Translates the source text.
        translated_data_list = self.translate_client.translate(values=source_text, target_language=to_language,
                                                               source_language=text_language, model=model)

        if isinstance(source_text, six.string_types):
            translated_data_list = [translated_data_list]

        # for debug purpose
        if debug_print:
            Tool.separate_debug_print_big(title="Start Translation Result")
            for cur_translated, i in zip(translated_data_list, range(translated_data_list.__len__())):
                Tool.separate_debug_print_small(title='Translate Line Number: ' + i.__str__())
                print(u'Text to translate: {}'.format(
                    cur_translated['input']))
                print(u'The translation: {}'.format(
                    cur_translated['translatedText']))
                if text_language is None:
                    print(u'Detected source language: {}'.format(
                        cur_translated['detectedSourceLanguage']))
                # translated_lines.append(cur_translated['translatedText'])
                Tool.separate_debug_print_small(title='Translation result end')
            Tool.separate_debug_print_big(title="Line Translation End")

        """
        The final returns if for the relevant
        result structure:
        just translation / translatedText / translatedText + input / all fields
        """
        if level == 0:
            # TODO: We Hve to decide how do we what the return value to be
            for cur_translated in translated_data_list:
                translated_lines.append(cur_translated['translatedText'])
            return translated_lines
        elif level == 1:
            for cur_translated in translated_data_list:
                translated_lines.append({'translatedText': cur_translated['translatedText']})
            return translated_lines
        elif level == 2:
            for cur_translated in translated_data_list:
                translated_lines.append({'translatedText': cur_translated['translatedText'],
                                         'input': cur_translated['input']})
            return translated_lines
        else:
            return translated_data_list

    def supp_language(self, language=None):
        """
        Lists all available languages and localizes them to the target language.
        :param language: The language to show the supported languages
        :return:
        """
        all_supported_language = self.translate_client.get_languages(target_language=language)

        Tool.separate_debug_print_big("Start the supported languages")
        for cur_language in all_supported_language:
            print(u'{name} ({language})'.format(**cur_language))
        Tool.separate_debug_print_big("End the supported languages")

    def language_detection(self, source_text, print_result=False):
        """
        Checks what is the language of the input text with probability
        :param source_text: The source text we want to detect the language
        :param print_result: To print the result, for the debug reason
        :return: The text language
        """

        language_result = self.translate_client.detect_language(source_text)

        # for debug reason
        if print_result:
            Tool.separate_debug_print_big(title='Start Detection Results')
            print('Text: {}'.format(source_text))
            print('Confidence: {}'.format(language_result['confidence']))
            print('Language: {}'.format(language_result['language']))
            Tool.separate_debug_print_big(title='End Detection Results')

        return language_result['language']


if __name__ == '__main__':

    # the temporary main
    test_text = "Hey, My name is Gidi. I like movies, sex, and Disco-Dancing. This is test for the translation"
    string_text = list()
    google_api = GoogleTranslateAPI()
    google_api.supp_language()

    string_text.append("this is text number one")
