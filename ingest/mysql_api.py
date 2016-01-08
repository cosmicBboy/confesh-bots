'''Module for loading data dumps'''

import pandas as pd
import mysql.connector
import os


def load_mysql(user, password, host, database, table):
    con = mysql.connector.connect(user = user,
                                  password = password,
                                  host = host,
                                  database = database)
    query = "SELECT * FROM %s" % table
    return pd.read_sql(query, con)


def mysql2csv(data, output_fp):
    data.to_csv(output_fp, index=False, encoding='utf-8')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user', type=str, required=True)
    parser.add_argument('-p', '--password', type=str, required=True)
    parser.add_argument('-host', '--host', type=str, required=True)
    parser.add_argument('-db', '--database', type=str, required=True)
    parser.add_argument('-t', '--table', type=str, required=True)
    parser.add_argument('-o', '--output_fp', type=str, required=True)
    args = parser.parse_args()

    data = load_mysql(args.user,
                      args.password,
                      args.host,
                      args.database,
                      args.table)

    mysql2csv(data, args.output_fp)
