PHONY: all

HOST=localhost
USER=root
INIT_CMD1=ALTER USER 'root'@'localhost' IDENTIFIED BY 'tmppass';
INIT_CMD2=CREATE USER '$(USER)'@'localhost' IDENTIFIED BY '';
INIT_CMD3=GRANT ALL PRIVILEGES ON * . * TO '$(USER)'@'localhost';
CHAR_SET=utf8

DATA_DUMP_FP=~/Downloads/*.sql

DB1=holyokecon
DB2=smithcon
OUTPUT_FP=./tmp
RAW=$(OUTPUT_FP)/raw
CLEAN=$(OUTPUT_FP)/clean

TABLES=confessional_secrets \
	   confessional_comments \
	   confessional_hashes \
	   confessional_reports \
	   confessional_codes

RAW_FILES=$(RAW)/*.csv

install-sql:
	brew install mysql

uninstall-sql:
	pkill -f mysql
	brew uninstall --force mysql
	brew cleanup

sql-init: sql-stop
	sudo rm -rf /usr/local/var/mysql
	mysqld --initialize-insecure
	mysql.server start
	mysql -u root --execute "$(INIT_CMD1) $(INIT_CMD2) $(INIT_CMD3)"

sql-stop:
	mysql.server stop

data-dump:
	for f in echo ${DATA_DUMP_FP}; \
	do \
		mysql -u ${USER} < $$f; \
	done

setup-pipeline:
	mkdir -p $(RAW) $(CLEAN)
	mkdir $(OUTPUT_FP)

requirements:
	pip2 install -r requirements.txt

sql2csv:
	for table in ${TABLES}; \
	do \
	    sql2csv --db mysql+mysqlconnector://${USER}@${HOST}/${DB1} \
	        --query "SELECT * FROM $$table" \
	        > ./${RAW}/${DB1}_$$table.csv; \
	    sql2csv --db mysql+mysqlconnector://${USER}@${HOST}/${DB2} \
	        --query "SELECT * FROM $$table" \
	        > ./${RAW}/${DB2}_$$table.csv; \
	done

preprocess-secrets:
	$(eval $FNAMES := $(shell echo ${RAW_FILES} | grep -o '[A-z]*_secrets\.csv')) \
	for file in $($FNAMES); \
	do \
		python preprocess.py -i ${RAW}/$$file -o ${CLEAN}/$$file \
							 --id id --raw confession --outcome comments ;\
	done

preprocess-comments:
	$(eval $FNAMES := $(shell echo ${RAW_FILES} | grep -o '[A-z]*_comments\.csv')) \
	for file in $($FNAMES); \
	do \
		python preprocess.py -i ${RAW}/$$file -o ${CLEAN}/$$file \
							 --id id --raw comment --fk_keys secret_id ;\
	done

preprocess-reports:
	$(eval $FNAMES := $(shell echo ${RAW_FILES} | grep -o '[A-z]*_reports\.csv')) \
	for file in $($FNAMES); \
	do \
		python preprocess.py -i ${RAW}/$$file -o ${CLEAN}/$$file \
							 --id id --raw reason --fk_keys secret_id comment_id ;\
	done

preprocess: preprocess-secrets preprocess-comments preprocess-reports

clean:
	rm -rf ${OUTPUT_FP}

# python preprocess.py -i './tmp/raw/holyokecon_confessional_secrets.csv' \
# 					 -o './tmp/clean/holyokecon_secrets_tokens.csv' \
# 					 -nrows 20
