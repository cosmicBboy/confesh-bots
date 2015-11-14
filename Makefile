PHONY: all

HOST=localhost
USER=`whoami`
INIT_CMD1=ALTER USER 'root'@'localhost' IDENTIFIED BY 'tmppass';
INIT_CMD2=CREATE USER '$(USER)'@'localhost' IDENTIFIED BY '';
INIT_CMD3=GRANT ALL PRIVILEGES ON * . * TO '$(USER)'@'localhost';

DATA_DUMP1=~/Dropbox/Confesh/data_dump/holyoke\ 10-16-15.sql
DATA_DUMP2=~/Dropbox/Confesh/data_dump/smith\ 10-16-15.sql

DB1=holyokecon
DB2=smithcon
OUTPUT_FP=./tmp

TABLES=confessional_secrets \
	   confessional_comments \
	   confessional_hashes \
	   confessional_reports \
	   confessional_codes

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
	mysql < $(DATA_DUMP1)
	mysql < $(DATA_DUMP2)

setup-pipeline:
	mkdir tmp

sql2csv: setup-pipeline
	for table in ${TABLES}; \
	do \
	    sql2csv --db mysql+mysqlconnector://${USER}@${HOST}/${DB1} \
	        --query "SELECT * FROM $$table" \
	        > ./${OUTPUT_FP}/${DB1}_$$table.csv; \
	    sql2csv --db mysql+mysqlconnector://${USER}@${HOST}/${DB2} \
	        --query "SELECT * FROM $$table" \
	        > ./${OUTPUT_FP}/${DB2}_$$table.csv; \
	done

clean:
	rm -rf ${OUTPUT_FP}

