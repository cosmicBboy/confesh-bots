.PHONY: clean

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

