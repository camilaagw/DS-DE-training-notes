# When to use RDDs
1. You want low-level transformation and actions and control on your dataset;
2. Your data is unstructured, such as media streams or streams of text;
3. You want to manipulate your data with functional programming constructs than domain specific expressions;
4. You don’t care about imposing a schema, such as columnar format while processing or accessing data attributes by name or column; and
5. You can forgo some optimization and performance benefits available with DataFrames and Datasets for structured and semi-structured data.

Taken from: https://databricks.com/fr/glossary/what-is-rdd

