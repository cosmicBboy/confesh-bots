PHONY: all

HOST=localhost
USER=root
INIT_CMD1=ALTER USER 'root'@'localhost' IDENTIFIED BY 'tmppass';
INIT_CMD2=CREATE USER '$(USER)'@'localhost' IDENTIFIED BY '';
INIT_CMD3=GRANT ALL PRIVILEGES ON * . * TO '$(USER)'@'localhost';

DATA_DUMP_FP=~/Downloads/*.sql

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
	for f in echo ${DATA_DUMP_FP}; \
	do \
		mysql -u ${USER} < $$f; \
	done

setup-pipeline:
	mkdir $(OUTPUT_FP)

requirements:
	pip2 install -r requirements.txt

sql2csv:
	for table in ${TABLES}; \
	do \
	    sql2csv --db mysql://${USER}@${HOST}/${DB1} \
	        --query "SELECT * FROM $$table" \
	        > ./${OUTPUT_FP}/${DB1}_$$table.csv; \
	    sql2csv --db mysql://${USER}@${HOST}/${DB2} \
	        --query "SELECT * FROM $$table" \
	        > ./${OUTPUT_FP}/${DB2}_$$table.csv; \
	done

clean:
	rm -rf ${OUTPUT_FP}

