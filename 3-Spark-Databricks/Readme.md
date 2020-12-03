# Spark Databricks Training

## Context
Hadoop: Map Reduce + HDFS. Disadvantage, map-reduce resilience imply a lot of IO disk: Ram->Disk->Ram. 
Solution: Decoupling Map Reduce from HDFS and introducing YARN: Allows Spark to run on top of HDFS. 
Yarn can be used in two modes: client (developer) or cluster (production).

Spark: In memory framework with several components: Spark SQL, Spark MLlib, Spark ML, Spark Streaming, and GraphX”.
Spark computations are represented as lineage graphs 

## RDDs
Spark is built around a data abstraction called Resilient Distributed Datasets (RDDs). 

RDDs are a representation of 
- lazily evaluated (computes transformations only when the final RDD data needs to be computed, i.e. when an action is called), 
- statically typed, 
- immutable (Transforming an RDD results in a new one), 
distributed collections of objects.  

RDDs objects are grouped into partitions which are stored in the executors.  An executor is a JVM holding a bunch of objects in memory

_Why are RDDs resilient?_ Immutability of RDDs allow Spark to rebuild an RDD from the previous RDD in the pipeline if there is a failure. 

## Dataframes
DataFrames (DF)are Distributed collection of objects of type Row, which can hold various types of tabular data.
Queries on dataframes are optimized by the catalyst optimiser, which is based on functional programming constructs in Scala

## Types of functions:
### Transformations
Take an RDD/DF as input and return an RDD/DF as output. Lazily evaluated. Examples: map, filter. Types of transformations:
 - Narrow: No shuffling. Depend on an unique and finite subset of parent partitions. Can be executed without info on other partitions (i.e. filter, coalesce, map)
 - Wide: Shuffling, more than one parent(i.e. sort, reduceByKey).

_Note: `join` can be either wide or narrow._

### Actions 
Return something else than an RDD/DF. Eagerly evaluated. Examples: Count, collect, write

### Others
Eager Drivers: (Print)
In Between: sortByKey

## Memory management:
Executor memory divided into two parts: Execution and Storage. If execution data is spilled to disk the performance of the job
is reduced Spark has to evict cache data when the executor memory is full and there is no more space for Storage;
one alternative is to spill to disk cached data. Recommendation: Evict storage, not execution

## Caching
To avoid unnecessary recalculations on the lineage graph one can persist (or cache) the data. Cache() is the same as Persist()
but with the default storage level MEMORY_ONLY, as regular Java objects. 
- Pros: Saves time when transformations are too complex and one has iterative algorithms 
- Cons: If cached data has to be spilled to disk, recomputing the lineage graph could be less expensive than reading the cached data from disk
     
Persist allows other options such as:
    - Disk spilling
    - Caching data as serialized Java objects (more compression but more time in serializing/deserializing ) 
    - Replicating cached data  
        
If cached data is persisted only in memory, some cached data will be lost when storage memory is full.
If cached data evicts execution data, job performance will suffer

Note: Excessive caching may cause overhead in the garbage collector

Tricks: For caching it is not recommended to use the option “only memory”, it is dangerous as the evicted data won’t be
spilled to disk. When you cache, materialize the cache with a count() (neither cache() nor persist() is an action.,
so the data is only cached during the first action call)

## Serializers: 
- SerDe: Serializer/Deserializer 
- Kryo:  10 times better than SerDe (has limited types) 
- Tungsten: compresses the data as binary format. Is a SerDe for Dataframes. Note: to measure real size of data in the RAM look at underlying RDD

## Jobs/Stages/Tasks

**Jobs**: An action will define a job. Every job will need a starting stage. Exceptions: Even though read is a transformation
it can cause jobs due to 1) schema inferring, and 2) column names reading (Spark needs to touch the data). Also, orderby()
(transformation) causes a job as it repartitions de data and the range partitioner has to access the data. 

**Stages**: Every wide operation will define a new stage Tasks: The number of tasks relates to the number of cores

### Analyzing jobs […]:

In the DAG: an “Exchange” implies a shuffle Duration is cpu time. To check for skewness in data partitioning, check for uniformity
in the distribution of values among tasks.

## Partitioning

Rule of thumb: Size of partition is between 50 and 200 MB. By default number of partitions is 200. When a transformation repartitions
 the data (i.e. `distinct()`, `groupby()`), by default it will produce 200 partitions. 
We can set this value to a lower number by using  `spark.sql.shuffle.partitions`. Ideally, the number of partitions should be a multiple
of the number of cores (2X is recommended)
- Increasing number of partitions -> Use Repartition (wide operation) 
- Decreasing number of partitions -> Use Coalesce as it requires less shuffling 

Note: Coalesce is not optimal when we have data skewed towards few repartitions.

When reading Data from Relational Database one needs to specify the number of pipes in order to not to have a single partition.
Two types of partitioners: 
- Hash partitioner (default)
- Range partitioner: Defines optimal number of partitions based on a range. 

Note that Highly skewed partitions often cause cached data to not to fit in memory.

## Optimization tricks:

- Replace `Join` by `Union + Distinct` (Union shuffles a lot less the data than join) 
- When joining two tables that significantly differ in size, broadcast the smallest one (up to 60 MB). (i.e. if df2 is small we can do `df1.join(broadcast(df2)`, “ID”)) 
- Inferring schema from JSON is faster than from CSV.
- When dealing with formats different from parquet, load the data and save it as parquet. Then load the parquet data. 
Parquet has the advantage that is a columnar data format and therefore it is better for dealing with column-oriented operations. 
Besides data will consume less space and the schema is for free. 
- Avoid zigzagging between RDDs and DF

## Catalyst

Optimizes dataframe queries for you. The catalyst creates multiple logical plans, then defines the physical plans, and then passes
them through the cost optimizer.

Catalyst phases: 
(1) analyzing a logical plan to resolve references, 
(2) logical plan optimisation, 
(3) physical planning, and
(4) code generation to compile parts of the query to Java bytecode. 

UDFs can not be optimized by the catalyst: Solution => add rules to catalyst 

## UDFs

Can be registered in spark to be used through the SQL api Pandas UDFs (in Spark 2.3). This improves performance usability of UDFs
in python trough Apache Arrow

## Distributed ML

**Linear Regression**: Analytical solution implies matrix inversion. After 4096 columns we can not invert matrices, when limit is reached gradient descent is used by using MSE of or hubber loss (epsilon 1.35)

**PCA**: It has limitations. When one can not use it, it is recommended to us SVD.

**Regulatization** Elastic net is supported in ML spark (you can switch completely to Rigde or Lasso) 

**Deep learning**: https://docs.databricks.com/applications/deep-learning/deep-learning-pipelines.html#deep-learning-pipelines, https://github.com/Azure/mmlspark/blob/master/notebooks/samples/DeepLearning%20-%20BiLSTM%20Medical%20Entity%20Extraction.ipynb

**Distributed ML with Sk-Learn**: Train multiple models on different nodes 

**ML Pipelines**: Spark ML is inspired in Sk-Learn. There we no significant new releases in the last years due to the focus on the DataFrame API.
Two main block types: 
- Estimators: (DF => Transformer ) Take dataframes as input and produce transformers (for instance, a model is a transformer). Used to train ML models. Use the method fit(). 
- Transformers: (DF => DF) Take dataframes as input and produce new dataframes with one or more columns appended to it (for instance, generate new features, predictions). Use the method transform(). 

ML pipelines were constructed as estimators. Recommended book: Spark: "The Definitive Guide". Recommended section: "Write your own estimator"