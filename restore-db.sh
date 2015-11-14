# A Module for converting sql to csv

read -p "DB: " DB
DB1=holyokecon
DB2=smithcon
HOST=localhost
OUTPUT_FP=./tmp

T1=confessional_secrets
T2=confessional_comments
T3=confessional_hashes
T4=confessional_reports
T5=confessional_codes

for table in ${T1} ${T2} ${T3} ${T4} ${T5}
do
    sql2csv --db mysql+mysqlconnector://`whoami`@${HOST}/${DB}\
        --query "SELECT * FROM ${table}"\
        > ./${OUTPUT_FP}/${DB}_$table.csv
done
