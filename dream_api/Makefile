PHONY: all

DBNAME='confesh-db'
COLLECTION='confession'
DREAM_FP='../data/dream_corpus_complete.csv'
MODEL_FP='./models/dream_bot_v'

run-dream-bot:
	python -W ignore dream_bot.py -db $(DBNAME) \
								  -c $(COLLECTION) \
								  -m $(MODEL_FP)$(VERSION) \
								  -dr $(DREAM_FP)

clean-model:
	rm $(MODEL_FP)$(VERSION)
