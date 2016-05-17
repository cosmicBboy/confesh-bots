.PHONY: clean

DBNAME=confesh-db
COLLECTION=confession
DATA_DIR=./tmp
REC=$(DATA_DIR)/recommendations.csv
PROCESSED_REC=$(DATA_DIR)/processed_recommendations.csv
MODEL=model3

# Default Communities
TRAIN_COMM=www
TARGET_COMM=bots
QUERY_COMM=www

# Default Parameters
PARAMS=-db $(DBNAME) \
	   -cl $(COLLECTION) \
	   -m $(MODEL) \
	   --rec_fp $(REC) \
	   --p_rec_fp $(PROCESSED_REC) \
	   --train_community $(TRAIN_COMM) \
	   --target_community $(TARGET_COMM) \
	   --query_community $(QUERY_COMM)

# Cron Build
CRON_1=* * * * * cd /confesh-bots
CRON_2=make -C thread_api run-thread-bot >> /threadbot_log.txt
CRON_3=tail -n 5000 /threadbot_log.txt > /ttmp.txt
CRON_4=cat /ttmp.txt > /threadbot_log.txt; rm /ttmp.txt
CRON=$(CRON_1); $(CRON_2); $(CRON_3); $(CRON_4)

clean:
	rm ./*.pyc
	rm ./tmp/*.csv

mongo-creds:
	echo "domain = <domain_creds>" > mongo_creds.py
	echo "port = <port>" >> mongo_creds.py

bitly-creds:
	echo "access_token = '<access_token>'" > bitly_creds.py

thread-creds:
	echo "auth_url = '<auth_url>'" >> threadbot_creds.py
	echo "post_url = '<post_url>'" >> threadbot_creds.py
	echo "post_comment_url = '<post_comment_url>'" >> threadbot_creds.py
	echo "passphrase = '<password>'" >> threadbot_creds.py

setup-pipeline: mongo-creds thread-creds bitly-creds
	mkdir $(DATA_DIR)

run-thread-bot:
	# Model has to be specified at the commandline
	python -W ignore thread_cli.py $(PARAMS)

echo-crontab-build:
	@echo '$(CRON)'

crontab-build:
	(crontab -l ; echo '$(CRON)') | crontab

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
