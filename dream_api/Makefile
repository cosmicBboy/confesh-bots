PHONY: all

DBNAME=confesh-db
COLLECTION=confession
MODELS_DIR=./models
DATA_DIR=./data
DREAM_FP=./data/dream_corpus_complete.csv
MODEL_FP=./models/dream_bot_v
DREAM_LOG=./dream_log.txt

run-dream-bot:
	python -W ignore dream_bot.py -db $(DBNAME) \
								  -c $(COLLECTION) \
								  -m $(MODEL_FP)$(VERSION) \
								  -dr $(DREAM_FP) \
								  --id_log $(DREAM_LOG)

setup-pipeline:
	mkdir $(DATA_DIR)

scrape-dreams:
	python scrape_dreams.py

parse-dreams:
	python parse_dreams.py -i ./raw -o $(DATA_DIR)/dream_corpus.csv

preprocess-dreams:
	python preprocess_dreams.py -i $(DATA_DIR)/dream_corpus.csv \
								-o $(DATA_DIR)/dream_corpus_complete.csv

build-pipeline: parse-dreams preprocess-dreams

clean-model:
	rm $(MODEL_FP)$(VERSION)

docker-build:
	sudo apt-get update
	sudo apt-get install apt-transport-https ca-certificates git tree docker.io

crontab-build:
	echo 'SHELL=/bin/bash' | crontab
	(crontab -l ; echo 'HOME=/') | crontab
	(crontab -l ; echo 'PATH=$(PATH)') | crontab
	(crontab -l ; echo 'MAILTO=""') | crontab
	(crontab -l ; echo '') | crontab
	(crontab -l ; echo "* * * * * cd /home/ubuntu/confesh-bot; make -C dream_api run-dream-bot VERSION=2.5 >> /home/ubuntu/dream_b\
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
