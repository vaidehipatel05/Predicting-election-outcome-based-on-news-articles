# Predicting-election-outcome-based-on-news-articles

We have taken Current news data to predict election outcome data and put that data onto Kafka cluster after that we consume that data and store that data onto Amazon S3. Crawl that data to build a glue catalog and analyze that data using Amazon Athena using SQL.

Fetching News API data from https://currentsapi.services/en

Ran the below commmand on SSH to verify if we are getting the news,
curl -X GET "https://api.currentsapi.services/v1/latest-news" -H "accept: application/json" -H "Authorization: API_TOKEN"

Created EC2 instance. Generated key and used below command to connect to EC2 via cli,
ssh -i "Kafka-news-election-project.pem" ec2-user@ec2-54-176-63-70.us-west-1.compute.amazonaws.com

Connected to EC2 machine, 
<img width="468" alt="image" src="https://github.com/vaidehipatel05/Predicting-election-outcome-based-on-news-articles/assets/152042524/a6adc48a-abd1-4b50-9444-950552802fc8">

Downloaded Kafka using below command,
wget https://downloads.apache.org/kafka/3.6.2/kafka_2.12-3.6.2.tgz
--2024-04-14 04:07:53--  https://downloads.apache.org/kafka/3.6.2/kafka_2.12-3.6.2.tgz
Resolving downloads.apache.org (downloads.apache.org)... 88.99.208.237, 135.181.214.104, 2a01:4f9:3a:2c57::2, ...
Connecting to downloads.apache.org (downloads.apache.org)|88.99.208.237|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 113986658 (109M) [application/x-gzip]
Saving to: ‘kafka_2.12-3.6.2.tgz’
100%[===================================================================================================================================>] 113,986,658 8.86MB/s   in 14s    
2024-04-14 04:08:08 (7.75 MB/s) - ‘kafka_2.12-3.6.2.tgz’ saved [113986658/113986658]

Extracted kafka,
tar -xvf kafka_2.12-3.6.2.tgz

Installed Java,
sudo yum install java-1.8.0-openjdk

Started Zookeeper and Server on different terminals,
bin/zookeeper-server-start.sh config/zookeeper.properties
bin/kafka-server-start.sh config/server.properties

Created topic on new terminal,
bin/kafka-topics.sh --create --topic demo_testing2 --bootstrap-server 18.144.34.135:9092  --replication-factor 1 --partitions 1

Public IP fetched from EC2,
18.144.34.135

Started Producer:
bin/kafka-console-producer.sh --topic demo_testing2 --bootstrap-server 18.144.34.135:9092

Started Consumer:
bin/kafka-console-consumer.sh --topic demo_testing2 --bootstrap-server 18.144.34.135:9092


