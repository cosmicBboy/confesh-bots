PHONY: all

DBNAME=confesh-db
COLLECTION=confession
MODELS_DIR=./models
DATA_DIR=./data
DREAM_FP=./data/dream_interpretations.csv
MODEL_FP=./models/dream_bot_v
DREAM_LOG=./dream_log.txt

run-dream-bot:
	python -W ignore dream_bot.py -db $(DBNAME) \
								  -c $(COLLECTION) \
								  -m $(MODEL_FP)$(VERSION) \
								  -dr $(DREAM_FP) \
								  --id_log $(DREAM_LOG) \
								  $(DRY)

requirements:
	pip install -r ../requirements.txt

mongo-creds:
	echo "domain = <domain_creds>" > mongo_creds.py
	echo "port = <port>" >> mongo_creds.py

dream-auth-creds:
	echo "auth_url = <auth_url>" > dream_auth_creds.py
	echo "post_url = <post_url>" >> dream_auth_creds.py
	echo "post_comment_url = <post_comment_url>" >> dream_auth_creds.py
	echo "passphrase = <passphrase>" >> dream_auth_creds.py

setup-pipeline: mongo-creds dream-auth-creds
	mkdir $(DATA_DIR)

scrape-dreams:
	python scrape_dreams.py

parse-dreams:
	python parse_dreams.py -i ./raw -o $(DATA_DIR)/dream_corpus.csv

preprocess-dreams:
	python preprocess_dreams.py -i $(DATA_DIR)/dream_corpus.csv \
								-o $(DATA_DIR)/dream_interpretations.csv

build-pipeline: requirements setup-pipeline parse-dreams preprocess-dreams

clean-model:
	rm $(MODEL_FP)$(VERSION)

docker-build:
	sudo apt-get update
	sudo apt-get install apt-transport-https ca-certificates git tree docker.io

cron-setup:
	apt-get install postfix heirloom-mailx

crontab-build:
	echo 'SHELL=/bin/bash' | crontab
	(crontab -l ; echo 'HOME=/') | crontab
	(crontab -l ; echo 'PATH=$(PATH)') | crontab
	(crontab -l ; echo 'MAILTO="root"') | crontab
	(crontab -l ; echo '') | crontab
	(crontab -l ; echo "* * * * * cd /confesh-bots; make -C dream_api run-dream-bot VERSION=2.6 >> /dream_b\
ot_log.txt") | crontab
	(crontab -l ; echo '') | crontab

cron-start:
	@echo 'awakening dreambot...'
	cron -L15
	$(MAKE) cron-status

cron-status:
	ps aux | grep cron\ -L15

cron-stop:
	@ echo 'dreambot goes to sleep'
	kill `pidof cron`
	$(MAKE) cron-status

start: cron-setup crontab-build

