_**Notes from Big data specialization from Coursera: https://www.coursera.org/learn/big-data-analysis/home/welcome**_

# I Spark optimizations

## Partitioning:

- Default partitioner uses key hashing (Alternative: range partitioner) 
- Co-partitioned RDDs reduce the volume of the shuffle
- Co-located RDDs don’t shuffle
- Preserve the partitioner to avoid shuffles

RDDs are co-partitioned if they are partitioned by the same
known partitioner:
```
>>> rdd1 = sc.parallelize([1, 2, 3]).partitionBy(10)
>>> rdd2 = sc.parallelize([4, 5, 6]).partitionBy(10)
>>> rdd1.partitioner == rdd2.partitioner
True
```

Partitions are co-located if they are both loaded into memory on
the same machine
```
>>> rdd1 = sc.parallelize([1, 2, 3]).partitionBy(10)
>>> rdd1.cache()
>>> rdd2 = sc.parallelize([4, 5, 6]).partitionBy(10)
>>> rdd2.cache()
>>> rdd1.partitioner == rdd2.partitioner
True
```

Transformations preserving partitioning: 

- map(f, preservePartitioning=True)
- flatMap(f, preservePartitioning=True)
- mapPartitions(f, preservePartitioning=True)
- mapPartitionsWithIndex(f, preservePartitioning=True)
- mapValues(f)
- flatMapValues(f)

Example:
```
links = lines.map(parseNeighbors)
 .distinct()
 .groupByKey()
 .partitionBy(2)
 .cache()
ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0), preservePartitioning=True)

for iteration in xrange(n):
 	contribs = links.join(ranks).flatMap( lambda u: computeContribs(u[1][0], u[1][1]))
 	ranks = contribs.reduceByKey(add, numPartitions=links.getNumPartitions()) .mapValues(lambda rank: rank * 0.85 + 0.15)
ranks.collect()

>> links.getNumPartitions(), links.partitioner
(2, <pyspark.rdd.Partitioner at 0x7f1ffdb65050>)
>> ranks.getNumPartitions(), ranks.partitioner
(2, <pyspark.rdd.Partitioner at 0x7f1ffdb65050>)
>> links.partitioner == ranks.partitioner
True
```

Tips for optimization:
- Specify number of partitions when appropriate with partitionBy(n) or numPartitions argument
- Use known partitioner to reduce shuffles
- When creating an RDD from another, use `preservePartitioning=True` if the transformation does not change the keys

## Serialization: Transfer data and code

Serialization is the process of translating data structures or object state into a format that can be stored (for example, in a file or memory buffer, or transmitted across a network connection link) and reconstructed later in the same or another computer environment

Serializers: 
- Java – slow, but robust
- Kryo – fast, but has corner cases `conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")`

## Functions optimization

- Avoid initializing objects inside your functions. Ideally, initialise objects only once on the driver 
- Broadcast variables to reduce initialization overhead. Broadcasting distributes the object across executors
- To reduce the number of function calls use `rdd.mapPartitions(myfunc)` instead of `rdd.map(myfunc)`. For this, you 
will need to add a `for` loop inside `myfunc` and to replace `return` for `yield`:

```
rdd.mapPartitions(myfunc)
obj = sc.broadcast(SomeLongRunningInit())
def myfunc(row_iter):
	for row in row_iter:
	    yield obj.value.apply(row)
 ```

 
# Spark UDFs (Spark SQL)

Using UDFs forces serialization. What are the alternative options?
- pyspark.sql.functions
- Hive UDFs/UDAFs
- Scala or Java (Extend Spark SQL via UDFs/UDAFs)

Example: The conversion of IP addresses was implemented in a Java function called ip2int. What we have to do is to add the jar file with the compiled function to the Spark context, and then register a temporary function. 
There's a problem though. You can't use a UDF implemented in such a way in the dataframe API. 
The only option available is to register a temporary function and to use it in a SQL query.
```
spark.sql("CREATE TEMPORARY FUNCTION ip2int as
 'ru.mipt.udfs.IPToInt'")

table1.createOrReplaceTempView("table1")
table2.createOrReplaceTempView("table2")

query =
"""
 select * from (select ip2int(ip) as ip2int from table1) t1
 join
 (select ip2int(ip) as ip2int from table2) t2 on t1.ip2int = t2.ip2int
"""
spark.sql(query).count()
```

## Catalyst (Spark SQL)

Catalyst is the Spark SQL query optimizer:
- A general library for representing trees and applying rules to manipulate them
- Contains libraries specific to relational query processing (e.g., expressions, logical query plans)
- Contains several sets of rules that handle different phases of query execution

Execution plans are represented as trees:
- 4 types of plans: Parsed Logical Plan, Analyzed Logical Plan, Optimized Logical Plan, Physical Plan
-  Optimizations are transformations of trees. Ex. of transformations: Constant folding, Filter pushdown
- Use `spark.sql(query).explain(True)` to see the different plans:
    - Parsed logical plan
    - Analyzed logical plan
    - Optimized logical plan
    - Physical plan

Advice: Always look at the execution plan:
- Check filter pushdown. If there is no filter pushdown check configs for `spark.sql.parquet.filterPushdown` and `spark.sql.orc.filterPushdown`
- Check join physical plan
- Check pruned columns


## Optimizing joins (Spark SQL)

Join algorithms
- Nested loop join: Universal algorithm. Slow! O(n*m)
- Hash join: More efficient O(n + m). Only supports equi-join. Hash table uses memory (can’t be used for huge tables)
- Sort-merge join: Between hash join and nested join O(n * log(n) + m * log(m)). Can be used for huge tables. Supports different join conditions (not only equi-joins). Join keys must be sortable.
Can be slow when table size is small as it needs to shuffle and sort

Types of physical plans
- Broadcast join: Broadcasts smaller dataframe. It is fast as it does not shuffle or sort the data. Perform map side join
- Shuffle hash join: `spark.sql.join.preferSortMergeJoin` is False.  Needs to shuffle, but not sort. Used when one side is ”much smaller” than the other so that a hash map can be build. 
Will OOM if data is skewed. You can force it if you are sure
- Sort-merge join (default): `spark.sql.join.preferSortMergeJoin` is True. Default join implementation. Keys must be sortable
- Broadcast nested loop join (?)
- Shuffle nested loop join (?)

`spark.sql.autoBroadcastJoinThreshold = 10485760`
Configures the maximum size in bytes for a table that will be broadcast

`spark.sql.shuffle.partitions = 200`
Configures the number of partitions to use when shuffling data for joins or aggregations (Does not help with skewed data)

`repartition(numPartitions, *cols)`
Shuffles data uniformly (helps for large joins)

Broadcast hint
```
from pyspark.sql.functions import broadcast
df1.join(broadcast(df2), on="join_column")
```
What can happen if you try to use the broadcast hint with the huge table?
- Out of memory error on the executor: Broadcasting has to use the memory of the executor to store the table. If there is not enough memory the application will fail.
- Broadcast timeout: There is an option spark.sql.broadcastTimeout which is 300 seconds by default. If broadcasting takes more time the application will fail


## Persisting and checkpointing

Spark provides ways to store intermediate results: persist &
checkpoint. These methods, compute the whole lineage graph and store the object in memory/disk.
Persisting is unreliable (stores the objects in memory/disk), checkpointing is reliable (saves partitions to HDFS).
Checkpointing is slow, persisting may be slow.

Storage levels for persisting:
- MEMORY_ONLY
- MEMORY_ONLY_2: 
- MEMORY_AND_DISK
- MEMORY_AND_DISK_2
- DISK_ONLY
- DISK_ONLY_2

_Note: the storage levels with suffix `_2` provide some redundancy by providing some replicas on different nodes_

Persistance tips:
- Used MEMORY_ONLY if it fits
- Be aware that recomputing may be as fast as reading from disk
- Persist iterative algorithms (ML)

Checkpointing:
```
sc.setCheckpointDir("/hdfs_directory/")
df.rdd.checkpoint()
```
_Note: Spark only checkpoints RDDs_

When to make a checkpoint?
- Noisy cluster (lots of users compiting for resources)
- Expensive and long computations


## Memory management

Two kinds of memory:
- Execution: Shuffles, joins, sorts and aggregations
- Storage: Cache data

Tips:
- If execution memory is full – spill to disk
- If storage memory is full – evict LRU blocks
- It is better to evict storage, not execution
- If your app relies on caching, tune `spark.memory.storageFraction` and `spark.executor.memory`

```
spark.executor.memory = 1g
Amount of memory to use per executor process

spark.memory.fraction = 0.6
Fraction of heap space used for execution and storage

spark.memory.storageFraction = 0.5
Amount of storage memory immune to eviction
```

## Resource allocation

### Basic concepts

*Cluster manager/ Resource manager*: A program allocating containers to your executors (i.e. YARN). 

*Worker node*: Can start in one or more executors

*Executor*: a JVM, which can process several tasks (Multiple tasks are processed as separate JVM thread. This allows to reduce overhead on starting and initializing an executor). A JVM has a heap memory and an off heap memory.  So when the application master sends resource requests to the resource manager, it gets total memory, specified with spark.executor.memory, and adds space for all of heap storage. This space is configured with spark.yarn.executor.memoryOverhead option, and is about 10% of total memory by default.


*Architecture of an app developed for YARN*:  there is a master node, which contains the resource manager itself and the scheduler. There are also slave nodes, and each node contains the so-called node manager. Node manager is a Hadoop process, which communicates with the resource manager and allocates containers from a particular node. Leave resources for the node manager daemon and other auxiliary Hadoop daemons.
There is also an application master. The application master is a container, which communicates with your driver, and sends resource allocation requests to the resource manager. As the application master is a container, you have to leave resources to start it. 



### Tips: 
- Use multiple cores per executor to benefit from JVM parallelism. If you allocate only one core to each executor, you can't use the benefits of JVM threads.
However, it is not recommended to use too many cores per executor. Rule of thumb: use at  most five core per executor so that it can achieve full write throughput
- Leave resources for OS and YARN daemons (1 core and 1Gb per node)
- Leave resources for AM (1 container)
- HDFS throughput is bounded by 5 cores (rule of thumb)
- Leave resources for off-heap (use a factor of 0.9)


Example - Cluster with 4 nodes (one node has 16 cores and 64G of memory): 
- Initial number of cores: 64
- Subtract 1 core and 1G for the node manager and auxiliary daemons (60 renamining cores, each node now has 63G). 
- Fix each executor to 5 cores for the best HDFS throughput. That means that if you divide 60 cores available by 5 cores per executor you will get 12 executors.
- Leave 1 container for the application master. 
- 63G of memory node divided by 3 executors per node results in 21G per executor, but counting all the off-heap storage, this results in 19G (21*0.9)

```
pyspark --num_executors 11
 --executor-cores 5
 --executor-memory 19G
 ```

 ## Dynamic allocation

Dynamic allocation starts executors when needed and releases when not needed

Why dynamic allocation?
- MR uses short-lived containers for each task
- Spark reuses long-running containers for speed (This may cause under-utilization)

Dynamic allocation requires tuning (Shuffle results should be stored in an external service)
```
spark.dynamicAllocation.enabled = false
#Whether to use dynamic resource allocation

spark.dynamicAllocation.executorIdleTimeout = 60s
#If an executor has been idle for more than this duration, the executor will be removed

spark.dynamicAllocation.cachedExecutorIdleTimeout = infinity
# If an executor which has cached data blocks has been idle for more than this duration, the executor will be removed

spark.dynamicAllocation.minExecutors = 0
#Lower bound for the number of executors if dynamic allocation is enabled

spark.dynamicAllocation.maxExecutors = infinity
# Upper bound for the number of executors if dynamic allocation is enabled

spark.shuffle.service.enabled = false
# Enables the external shuffle service. You need this one for dynamic allocation.
```

More info: https://www.slideshare.net/databricks/dynamic-allocation-in-spark, https://www.slideshare.net/SparkSummit/spark-summit-eu-talk-by-luc-bourlier


## Speculative execution

Straggler: task which runs significantly slower than all the others. 

Causes for stragglers:
- Equal workload, unequal resources: Speculative execution may help
- Equal resources, unequal workload (Skew data causes false stragglers): Salting, repartition or custom partitioner might help
- Bugs

### Configuring speculative execution:
```
spark.speculation = false
# If set to "true", performs speculative execution of tasks

spark.speculation.interval = 100ms

spark.speculation.multiplier = 1.5

spark.speculation.quantile = 0.75
# Fraction of tasks which must be complete before speculation is enabled for a particular stage
```

### Example configuring `spark.sql.shuffle.partitions` and using `DISTRIBUTE_BY`
Taken from: https://bigdatacraziness.wordpress.com/2018/01/05/oh-my-god-is-my-data-skewed/

```
from pyspark.sql import SQLContext
from pyspark.sql Row
 
sqlcontext = SQLContext(sc)
rdd = sc.textFile('/user/cloudera/sampledata/test.txt')
rdd2 = rdd.map(lambda x:x.split(',')).map(lambda x: Row(key=x[o],value=x[1]))
dataframe = sqlcontext.createDataFrame(rdd2)
dataframe.registerTempTable("df")
sqlcontext.sql("SET spark.sql.shuffle.partitions = 5")
sqlcontext.sql("SELECT * FROM df DISTRIBUTE BY key, value")
```
Note : “Distribute By” doesn’t guarantee that data will be distributed evenly between partitions. It all depends on the hash of the expression by which we distribut

 ### Salting Technique
 
"Consider you have datasets consisting of 4 unique keys out of which 1st key is having 2000 records,
2nd key is having 500 records, 3rd and 4th key is having 50 and 30 records respectively. This indicates
if partitions is based on original key then we can observe imbalanced distribution of data across the partitions.
In order to curb this situation we should modified our original keys to some modified keys whose hash partitioning cause
the proper distributions of records among the partitions" Taken from: https://bigdatacraziness.wordpress.com/2018/01/05/oh-my-god-is-my-data-skewed/


# II Hive tips

Use:
- Partitioning, bucketing and sorting to improve performance query
- Built-in techniques to handle skewed data
- Built-in UDF (User Defined Functions), UDAF (Aggregate functions), UDTF (Table-generating functions)
- Columnar format for efficiencly space utilization (different compression algorithms can be applied to different types of data), fast data loading, fast query processing, strong adaptivity to highly dynamic workload patterns

Recall: The difference between managed and external tables is that DROP TABLE statement, in managed tables, 
will drop the table and delete table's data. Whereas, for external table DROP TABLE will drop only the table and data
will remain as is and can be used for creating other tables over it.