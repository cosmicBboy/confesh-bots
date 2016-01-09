"""
Module for getting http responses from

http://www.dreammoods.com/

author: Niels Bantilan
"""

import requests
from bs4 import BeautifulSoup


def create_query_string(domain, path_invar, path_var):
    return "{}/{}/{}".format(domain, path_invar, path_var)

def get_query(query_string):
    return requests.get(query_string)

def soupify(response):
    return BeautifulSoup(response, 'html.parser')

domain = "http://www.dreammoods.com"
path_invar = "dreamdictionary"
path_var = "c.htm"
response = get_query(create_query_string(domain, path_invar, path_var))
soup = soupify(response.text)

# Grab relevant dream html elements
style = "margin-left: 8; margin-right: 10; margin-bottom: 10px"
attrs = {
    "align": "left",
    "style": style,
}
ps = soup.find_all('p')#, attrs=attrs)

# Note: There are 6 empty paragraph tags before the term definitions start
for i, p in enumerate(ps):
    print i
    print p
    print ''
    print ''
    print ''

