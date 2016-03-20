"""
Module for parsing dreammoods html

http://www.dreammoods.com/

author: Niels Bantilan

TODO: use "TOP" as a dream entry delimiter
"""
import logging
import codecs
import string
import pandas as pd
from bs4 import BeautifulSoup
from collections import OrderedDict
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

FORMAT_STRING = 'dreams_{}.html'
TABLE_FORMAT = [u'html', u'td', u'tr', u'center', u'p', u'table', u'table']
STOP_WORDS = ['TOP']
EXCLUDE_VOCAB = [
    'to',
    'tosee',
    'please see',
    'if',
    'alternatively'
    'in particular',
    'that',
    'todream'
]

def parse_all_dreams(input_fp):
    all_dreams_dict = OrderedDict()
    all_dreams_list = [parse_dream(alpha, input_fp)
                       for alpha in string.ascii_lowercase]
    for ddict in all_dreams_list:
        all_dreams_dict.update(ddict)
    df = pd.DataFrame({"vocab": all_dreams_dict.keys(),
                       "definitions": all_dreams_dict.values()})
    return df[['vocab', 'definitions']]


def parse_dream(alpha, input_fp):
    dream_fp = "{}/{}".format(input_fp, FORMAT_STRING.format(alpha))
    logging.info(dream_fp)
    soup = read_html(dream_fp)
    result = soup.find_all(recursive=False)
    dream_format = [t.name for t in result]
    if dream_format == TABLE_FORMAT:
        result = parse_table_format(result, alpha)
    else:
        result = parse_paragraph_format(result, alpha)
    result = create_dream_corpus(result, alpha)
    return result


def parse_table_format(soup, alpha):
    soup = soup[-2].find('table').find_all('td', attrs={'width': '750'})
    soup = soup[0].find_all('p', recursive=False)
    result = [unicode(t.text).encode('ascii', 'ignore').strip()
              for t in soup if t.name in ['p', 'strong', 'font']]
    return prep_dream(result, alpha)


def parse_paragraph_format(soup, alpha):
    '''
    Function assumes that if the 5 element in soup iterable contains p tags
    where the alpha.upper() character appears twice, that means to include the
    soup[5] element in the result list.
    '''
    result = [unicode(t.text).encode('ascii', 'ignore').strip()
              for t in soup if t.name in ['p', 'strong', 'font']]
    tmp = soup[5].find_all('p')
    include_table = len([p.text for p in tmp if p.text == alpha.upper()]) == 2
    if include_table:
        index = [i for i, p in enumerate(tmp) if p.text == alpha.upper()][-1]
        prepend_result = [t.text for t in tmp][index: ]
        prepend_result.extend(result)
        result = prepend_result
    return prep_dream(result, alpha)


def create_dream_corpus(text_list, alpha, exclude_vocab=EXCLUDE_VOCAB):
    vocab = OrderedDict()
    current_vocab = None
    skip_word = ""
    for i, t in enumerate(text_list):
        if t == skip_word:
            skip_word = ""
            continue
        if is_vocab(t, alpha, exclude_vocab):
            current_vocab = t
            next_word = text_list[i + 1].strip()
            if next_is_part_of_vocab(next_word, exclude_vocab):
                current_vocab = " ".join([current_vocab, next_word])
                skip_word = next_word
            vocab[current_vocab] = []
        elif t in STOP_WORDS:
            continue
        else:
            vocab[current_vocab].append(t.strip())
    return prep_dream_corpus(vocab)


def prep_dream_corpus(vocab_dict):
    '''
    Input:
    A dictionary that assumes this structure:
    {'vocab': ['definition_text1', 'definition_text2'], ...}

    Output:
    A dictionary with this structure:
    {'vocab': 'definition', ...}
    '''
    dream_dict = OrderedDict()
    for k, v in vocab_dict.items():
        dream_dict[k.lower()] = " ".join(v).lower()
    return dream_dict


def is_vocab(text, alpha, exclude_vocab):
    if (text[0] == alpha.upper() and
        text.lower() not in exclude_vocab and
        text.split()[0].lower() not in exclude_vocab and
        "\"" not in text and "," not in text and "." not in text and
        len(text.split()) < 5):

        return True
    else:
        return False


def next_is_part_of_vocab(text, exclude_vocab):
    if (text.strip()[0] in string.uppercase and
        text.lower() not in exclude_vocab and
        text.split()[0].lower() not in exclude_vocab and
        "\"" not in text and "," not in text and "." not in text and
        len(text.split()) < 5):

        return True
    if ('of' in text.split()[0]):
        return True
    else:
        return False


def prep_dream(text_list, alpha):
    if text_list[0] == "":
        text_list[0] = alpha.upper()
    if 'page' in text_list[0].lower():
        text_list = text_list[1:]
    if text_list[1] == 'to':
        text_list = text_list[3:]
    if text_list[0] != alpha.upper():
        text_list.insert(0, alpha.upper())

    text_list = '\n'.join(text_list)
    text_list = text_list.split('\n')

    # Remove all elements after the occurrence of 'Page'
    # This is the simple algorithm we use to remove the extra text
    # at the end of a dream page that doesn't hold any dream vocabulary
    # or dream definitions
    try:
        page_index = text_list.index('Page')
        text_list = text_list[:page_index]
    except:
        pass

    text_list = [t for t in text_list if not t.strip() == ""]
    return text_list


def read_html(fp):
    with codecs.open(fp, "rb", "utf-8") as f:
        html = f.read()
        return soupify(unicode(html).encode('utf8'))


def soupify(response):
    return BeautifulSoup(response, 'html.parser')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', help='raw dreams filepath', type=str)
    parser.add_argument('-o', help='clean dreams filepath', type=str)
    args = parser.parse_args()

    dreams = parse_all_dreams(args.i)
    logging.info("\n{}".format(dreams.head()))
    dreams.to_csv(args.o, index=False, encoding='utf-8')
