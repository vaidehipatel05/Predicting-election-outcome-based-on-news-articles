from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType

# Create a Spark session
spark = SparkSession.builder \
    .appName("KafkaStreamReader") \
    .getOrCreate()

# Define the schema for the incoming Kafka message
schema = StructType() \
    .add("title", StringType()) \
    .add("description", StringType()) \
    .add("author", StringType()) \
    .add("url", StringType()) \
    .add("image", StringType()) \
    .add("language", StringType()) \
    .add("category", StringType()) \
    .add("published", StringType())

# Read messages from Kafka
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "demo_testing2") \
    .load()

# Convert the value column from binary to string
df = df.selectExpr("CAST(value AS STRING)")

# Parse the JSON data into columns using the defined schema
df = df.select(from_json("value", schema).alias("data")).select("data.*")

#print("****************************************************this is outptu in df", df)
# Write the streaming DataFrame to a Parquet file
query = df.writeStream.outputMode("append").format("csv").option("path", "/Users/vaidehipatel/Downloads/").option("checkpointLocation", "/Users/vaidehipatel/Downloads/").start()

# Write the streaming DataFrame to a single CSV file
#query = df.coalesce(1).writeStream.outputMode("append").format("csv").option("path", "/Users/vaidehipatel/Downloads/").option("checkpointLocation", "/Users/vaidehipatel/Downloads/").start()


query.awaitTermination()
