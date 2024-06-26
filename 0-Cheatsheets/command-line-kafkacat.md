Consume messages from a Kafka topic using kcat, formerly known as kafkacat:

kcat -C -s avro -r <REGISTRY-NAME> -t <TOPIC-NAME> -o -10

- `kcat -C`: The `kcat` command with the `-C` flag is used to consume messages from a Kafka topic.
- `-s avro`: The `-s` flag indicates the serialization format used by the topic. In this case, `avro` is the serialization format.
- `-r <REGISTRY-NAME>`: The `-r` flag is used to specify the URL of the Schema Registry. The Schema Registry is a service that provides a RESTful interface for storing and retrieving Avro schemas used by Kafka.
- `-t <TOPIC-NAME>`: The `-t` flag is used to specify the Kafka topic to consume messages from.
- `-o -10`: The `-o` flag is an optional setting to control the offset at which to start consuming messages. Here `-10` means that consumption starts with the last 10 messages (i.e., the last 10 messages will be repeated).
