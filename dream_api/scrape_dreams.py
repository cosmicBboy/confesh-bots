"""
Module for scraping entries from

http://www.dreammoods.com/

author: Niels Bantilan
"""

import requests
import string
import codecs
from time import sleep

DOMAIN = "http://www.dreammoods.com"
PATH_INVAR = "dreamdictionary"
EXTENSION = "htm"


def scrape_dreams():
    for alpha in string.ascii_lowercase:
        print "Scraping dreams starting with letter: '{}'".format(alpha)
        fp = "./corpus/dreams_{}.html".format(alpha)
        response = scrape_dream_char(alpha)
        print "Writing response to {}".format(fp)
        write_to_file(fp, response.text)
        sleep(0.1)


def scrape_dream_char(alpha_char):
    '''
    Returns html response. Trues to append _all to alpha_car to fetch all
    entries for a letter, else it fetches first page.
    '''
    try:
        path_var = "{}_all".format(alpha_char)
        query = create_query_string(DOMAIN, PATH_INVAR, path_var, EXTENSION)
        return requests.get(query)
    except:
        query = create_query_string(DOMAIN, PATH_INVAR, alpha_char, EXTENSION)
        return requests.get(query)


def create_query_string(domain, path_invar, path_var, extension):
    return "{}/{}/{}.{}".format(domain, path_invar, path_var, extension)


def write_to_file(fp, response):
    with codecs.open(fp, "w", "utf-8") as f:
        f.write(response)


if __name__ == "__main__":
    scrape_dreams()