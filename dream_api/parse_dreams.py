"""
Module for parsing dreammoods html

http://www.dreammoods.com/

author: Niels Bantilan
"""
import codecs
import string
import re
from bs4 import BeautifulSoup

FORMAT_STRING = 'raw/dreams_{}.html'
TABLE_FORMAT = [u'html', u'td', u'tr', u'center', u'p', u'table', u'table']

def parse_all_dreams():
    return [parse_dream(alpha) for alpha in string.ascii_lowercase]


def parse_dream(alpha):
    # print ">>>", alpha
    soup = read_html(FORMAT_STRING.format(alpha))
    result = soup.find_all(recursive=False)
    dream_format = [t.name for t in result]
    if dream_format == TABLE_FORMAT:
        result = parse_table_format(result, alpha)
    else:
        result = parse_paragraph_format(result, alpha)
    result = find_vocab(result, alpha)
    return result


def parse_table_format(soup, alpha):
    soup = soup[-2].find('table').find_all('td', attrs={'width': '750'})
    soup = soup[0].find_all('p', recursive=False)
    result = [unicode(t.text).encode('ascii', 'ignore').strip()
              for t in soup if t.name in ['p']]
    return prep_dream(result, alpha)


def parse_paragraph_format(soup, alpha):
    '''
    Function assumes that if the 5 element in soup iterable contains p tags
    where the alpha.upper() character appears twice, that means to include the
    soup[5] element in the result list.
    '''
    result = [unicode(t.text).encode('ascii', 'ignore').strip()
              for t in soup if t.name in ['p', 'strong']]
    tmp = soup[5].find_all('p')
    include_table = len([p.text for p in tmp if p.text == alpha.upper()]) == 2
    if include_table:
        index = [i for i, p in enumerate(tmp) if p.text == alpha.upper()][-1]
        prepend_result = [t.text for t in tmp][index: ]
        prepend_result.extend(result)
        result = prepend_result
    return prep_dream(result, alpha)


def find_vocab(text_list, alpha, exclude_vocab=['to', 'tosee']):
    vocab = [t for t in text_list if t[0] == alpha.upper()
             and t.lower() not in exclude_vocab
             and t.split()[0].lower() not in exclude_vocab
             and len(t.split()) < 5]
    return vocab


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


def find_terms():
    pass


if __name__ == "__main__":
    dreams = parse_all_dreams()
    for i, a in enumerate(string.ascii_lowercase):
        print "<<<", a
        print len(dreams[i])
        print unicode(dreams[i]).encode('utf-8')
