# Confesh

All things confesh

# Data Analysis

- Understand how queer, gender issues (trans) topics and analysis on how sentiments has evolved over time.
- Seeking to understand the topics of conversation over time by community.
- Seeking to understand positive/negative sentiments in regards to topic over time by community.

# Prerequisites

To install project requirements, run from the confesh root:
```
make requirements
make install-sql
make setup-pipeline
```

# Confesh Datadump

To dump confesh data into your local mysql server, we assume that you have
access to data and you can download the files in your `~/Downloads` folder.
Once you have downloaded the .sql data dumps, you can run

```
make data-dump
make sql2csv

# Check if you have all the csv files:
ls ./tmp

# holyokecon_confessional_codes.csv
# holyokecon_confessional_comments.csv
# holyokecon_confessional_hashes.csv
# holyokecon_confessional_reports.csv
# holyokecon_confessional_secrets.csv
# smithcon_confessional_codes.csv
# smithcon_confessional_comments.csv
# smithcon_confessional_hashes.csv
# smithcon_confessional_reports.csv
# smithcon_confessional_secrets.csv
```
