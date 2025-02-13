import re

import yake
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_ru')

from .consts import remove_chars, replace_chars, flt_chars


def get_excluded_list():
    stop_words = (
            nltk.corpus.stopwords.words('russian')
            +
            nltk.corpus.stopwords.words('english')
            +
            nltk.corpus.stopwords.words('azerbaijani')
    )
    exclude = [x for x in set(stop_words + flt_chars) if len(x) > 0] + ['pet']
    exclude.remove('up')
    exclude.remove('s')
    exclude.remove('ya')
    return set(exclude)


def simple_process_item(x: str, exclude: list):
    '''
    Lowercase, remove and replace charecters in original string x
    Remove double spaces after cleaning
    '''
    x = x.lower()
    for char in remove_chars.keys():
        x = x.replace(char, ' ')
    for k, v in replace_chars.items():
        x = x.replace(k, v)
    item = ' '.join(re.split(r'(\d+)', x))
    while item.count(2 * " ") > 0:
        item = item.replace(2 * " ", " ")
    item = item.replace(' .', '.').replace('. ', '.')
    return ' '.join([x for x in item.split() if x not in exclude])


def count_common_words(str1, str2):
    '''Method for getting score based on amount of full words intersecting in two strings'''
    set1 = set([x for x in str1.lower().split() if len(x) > 3 and not x.isdigit()])
    set2 = set([x for x in str2.lower().split() if len(x) > 3 and not x.isdigit()])
    if ((len(set1) + len(set2))/2) == 0:
        return 0
    return len(set1.intersection(set2))/((len(set1) + len(set2))/2)


def count_common_digits(str1, str2):
    '''Method for getting score based on amount of numbers intersecting in two strings'''
    set1 = set([x for x in str1.lower().split() if x.isdigit()])
    set2 = set([x for x in str2.lower().split() if x.isdigit()])
    if ((len(set1) + len(set2))/2) == 0:
        return 0
    return len(set1.intersection(set2))/((len(set1) + len(set2))/2)


def count_digit_share(x: str):
    '''Method for getting share of numbers in the length of string'''
    counter = 0
    for char in x:
        if char.isdigit():
            counter += 1
    return counter/len(x.replace(' ', '')) if len(x) > len(x.replace(' ', '')) and counter > 0 else 0.5


def remove_numbers(text):
    '''
    Method remove all numbers from string and returns 3 first words
    Words in final string filtered by length - 2 and custom stop words
    '''
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split('.'))
    return ' '.join([x for x in text.split() if len(x) > 2 and x not in ('pct', 'dad')][:3])


# Extractor of key words for parent and product groups names extracting
kw_extractor = yake.KeywordExtractor(
    n=2,
    dedupLim=0.5,
    dedupFunc='sequencematcher',
    windowsSize=2,
    top=3,
    features=None
)


def get_product_group(word: str):
    '''Use YAKE to extract product group'''
    keywords = kw_extractor.extract_keywords(word)
    x = {kw[0]: kw[1] for kw in keywords}
    if len(x) > 0:
        return min(x, key=x.get)
    else:
        return ''


def get_parent_group(words: str):
    '''Use YAKE to extract parent group'''
    if len(words) > 0:
        fltr = kw_extractor.extract_keywords(words)
        fltr_dct = {kw[0]: kw[1] for kw in fltr}
        return max(fltr_dct, key=fltr_dct.get)
    else:
        return ''
