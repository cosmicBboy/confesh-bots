# A Module for converting sql to csv

read -p "Username: " SQL_USER
read -s -p "Enter password: " SQL_PW
read -p "DB: " DB
HOST=localhost
OUTPUT_FP=./tmp

T1=confessional_secrets
T2=confessional_comments
T3=confessional_hashes
T4=confessional_reports
T5=confessional_codes

for table in ${T1} ${T2} ${T3} ${T4} ${T5}
do
    sql2csv --db mysql+mysqlconnector://${SQL_USER}:${SQL_PW}@${HOST}/${DB}\
        --query "SELECT * FROM ${table}"\
        > ./${OUTPUT_FP}/${DB}_$table.csv
done
