PHONY: all

DBNAME=confesh-db
COLLECTION=confession
MODELS_DIR=./models
DREAM_FP=../data/dream_corpus_complete.csv
MODEL_FP=./models/dream_bot_v
DREAM_LOG=./dream_log.txt

run-dream-bot:
	python -W ignore dream_bot.py -db $(DBNAME) \
								  -c $(COLLECTION) \
								  -m $(MODEL_FP)$(VERSION) \
								  -dr $(DREAM_FP) \
								  --id_log $(DREAM_LOG)

setup-pipeline:
	mkdir $(MODELS)

clean-model:
	rm $(MODEL_FP)$(VERSION)

docker-build:
	sudo apt-get update
	sudo apt-get install apt-transport-https ca-certificates git tree docker.io

crontab-build:
	echo 'SHELL=/bin/bash' | crontab
	(crontab -l ; echo 'HOME=/') | crontab
	(crontab -l ; echo 'PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin') | crontab
	(crontab -l ; echo '') | crontab
	(crontab -l ; echo "* * * * * cd /; echo 'hello world' >> ./crontest.txt") | crontab
	(crontab -l ; echo '') | crontab
