PHONY: all

DBNAME=confesh-db
COLLECTION=confession
DREAM_FP=../data/dream_corpus_complete.csv
MODEL_FP=./models/dream_bot_v
DREAM_LOG=./dream_log.txt

run-dream-bot:
	python -W ignore dream_bot.py -db $(DBNAME) \
								  -c $(COLLECTION) \
								  -m $(MODEL_FP)$(VERSION) \
								  -dr $(DREAM_FP) \
								  --id_log $(DREAM_LOG)

clean-model:
	rm $(MODEL_FP)$(VERSION)
