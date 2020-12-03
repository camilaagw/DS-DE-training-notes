# What's new in Spark 3

## Adaptive query optimization
Uses statistics from runtime to optimise physical plan. As it is still on the experimental phase, it needs to be enabled.

Options to tune:
- `spark.sql.adaptative.enabled`
- `spark.sql.adaptative.skewedJoin.enabled`
- `spark.sql.adaptative.localShuffleReader.enabled`
- `spark.sql.adaptative.coalescePartitions.enabled`

## Dynamic partitioning pruning
In standard database pruning means that the optimizer will avoid reading files that cannot contain the data that you are looking for
In particular, we consider a star schema which consists of one or multiple fact tables referencing any number of dimension tables. In such join operations, we can prune the partitions the join reads from a fact table by identifying those partitions that result from filtering the dimension tables.
Needs data partitioned by a proper column (normally a column on which there is a filtering). Enabled by default.

## Join hints
Hint to the catalyst which join to use:
```markdown
df1.hint("SHUFFLE_MERGE").join(df2, df1.var1 == df2.var2).show()
```
Ideally not needed with adaptayive query optimization enabled.
