# I. Recap of main ML algorithms

**Linear models**
 Split space into 2 subspaces with a hyperplane (Ex.: Linear Reg., Logistic Reg., SVM). Good for sparse, high dimensional data, where there are linear dependencies.
 *Logistic regression assumption: Data samples are independently and identically distributed (so-called i.i.d assumption)*

**Tree-based methods**
Decision tree is the main construction block. Uses divide and conquer approach to recursively divide the space into subspaces (or boxes). In regression tasks aproximates points in the boxes with a constant. A bad choise when data has linear dependencies.

**kNN-based methods**
Points close to each other are likely to have the same labels. Heavily rely on how to measure points "closeness"

**Neural Networks** 
Produce smooth separating curve (in contrast to decision trees).

***No free lunch theorem***: *There is no method which outperforms all others for all tasks. There is no "silver bullet algorithm"*

# II. Feature processing

Remember: Preprocessing and generation pipelines depend on a model type.

For instance:
- Lineal models require hot-encoded variables if the dependencies between numerical features and target are not linear
- Decision-tree-based models do not need one hot encoding. These models also don't depend on feature scaling


## A. Numeric features

### Preprocessing

#### Feature scaling:
Scale all features to one a specific range/distribution, so that their  impact on the model will be roughly similar.

Tree-based models don't depend on feature scaling while Linear Models , kNN and neural networks do: regularization impact is proportional to feature scale and  gradient descent methods will go crazy without the proper scaling.

Types: 
- Min-max scaling (distributions do not change)
- Standar scaler (Goal: $mean=0$ and $std=1$)


#### Outliers treatment
Outliers affect linear models. To prevent them we can use winzorisation: clip feature values ammong a low and a high percentile.

#### Rank transformation
Used to set spaces between sorted feature values to be equal. Can be also used to treat outliers as with the rank they will be closer to other values. Linear models, KNN and Neural Networks can also benefit from this transformation. Can be applied to train+test data.

#### Other transformations
The following transformations drive big values to the features average value. Besides, values closer to zero become more distinguishable :
- Log transform: `np.log(1+x)`
- Raising to the power <1: `np.sqrt(x+2/3)`

*Hint: use ensemble methods with different preprocessings to achieve better results*

### Features generation
Examples:
- **Interactions**: click rate, price per $m^2$, calculated distances with Pytagoras theorem. In adition to Linear Models, these are beneficial for methods such as GBT as they experience dificultiexs with approximations of multiplications and divissions
- **Fractional part**. Fom 5.99, 0.99 is the factional part. Useful for price variables, for instance


## B. Categorical and ordinal features

#### Label encoding
Map each category to an unique value. Useful for tree-based models. Not always useful for linear models or NN unless the values assigned are correlated with the target. 

Types:
- Alphabetical order
- Order or apearance
- Meaningful order (in case the categorical variable is ordinal)
- Frequency encoding: Map each category to its frequecy in the data set. For multiple categories with the same frequency we can use a rank operation 

#### One-hot encoding
Creates a new column for every category. Works better with non-tree-based models than with tree-based ones due to the explosion in the number of features


## C. Datetime and coordinates

#### Datetime 

1. Periodicity: Day number in week, month, season, year, second, minute, hour.
2. Time since
	a. Row-independent moment: For example, since 00:00:00 UTC, 1 January 1970;
	b. Row-dependent important moment: Number of days left until next holidays/ time passed after last holiday.
3. Difference between dates: datetime_feature_1 - datetime_feature_2

#### Coordinates

Ideas:
- Calculate distance to the nearest shop, hospital, school, etc
- Organize data into clusters and add distance to the center of clusters
- Compute aggregated stats per area: Price per $m^2$, number of flats
with tree based models, you can add rotations of the coordinates as new features

## D. Handling missing values

Different methods to fill the missing vaues: 
- Replace them by other value (999, -1) (decision-based models)
- Mean, median  (linear models, NN,  KNN)
- Recontruct values (time series)
- Add Isnull feature

Note 1 : Avoid filling missing values before feature generation 
Note 2: Finding hidden missing values encoded as a numerial value: Plot histogram and look for abnormal peaks!
Note 3: XgBoost can handle NaN

# III. Exploratory data analysis

General advice: 
- Get domain knowledge
- Check if the data is intuitive and consistent with its definition
- It is crucial to understand how the data was generated to set up a proper validation schema. For instance if the distribution of the test set is different from the training set, we cannot validate our model with a sample of the training set
- Visualization is important: Patterns lead to questions (or hypothesis) and hypothesis can be supported by graphics

Useful Pandas methods: `df.types`, `df.info()`, `df.describe()`, `x._value_counts()`, `x.isnull()`

Useful visualization tools:
- Histograms (be careful with bins!):  `plt.hist(x)`
- Plot index vs value (can show leakages): `plt.plot(x,'.')` 
- Plot index vs value colored by target value: `plt.scatter(range(len(x)),x ,c=y)`
- Exploring features relations: `plt.scatter(x1,x2)`, `pd.scatter_matrix(df)`
- Correlation matrices: `plt.matshow()` (TODO: add clustering example)
- Plot mean values of individual features: `df.mean().sort_values().plot(style='.')

**Dataset cleaning and other aspects to check**

- Remove features with zero variance in training set
- Check for dupllicated rows (check if same rows have same label)
- Check for duplicated columns:
```
for col in categorical_cols:
	train_test[col] = train_test[col].factorize()[0]

train_test.T.drop_duplicates
```

# IV. Validation strategies

There are three main validation strategies:
1. **Holdout** (i.e. 70% for training, 30% for testing). Use if validation results and/or optimal parameters among different splits do not change "too much"
2. **k-Fold**. Use if scores and optimal parameters differ for different splits.
3. **Leave One Out** (iterate though every point in your data). Use if you have too little data.

*Hint: Stratification preserves the same target distribution over different folds*

Data slitting stategies:
- Random-based splitting
- Time-based splitting. Special case: Window moving validation.  
- ID-based splitting

*Note: Ensure similar distributions between your validation set and the test set (In most cases, the reality!)*

### Specific advice for Kaggle competitions: 

 If we have big dispersion of scores on validation stage, we should do extensive validation:
- Average scores from different KFold splits
- Tune model on one split, evaluate score on the other

If submission’s score do not match local validation score,we should
- Check if we have too little data in public LB
- Check if we overfitted
- Check if we chose correct splitting strategy
- Check if train/test have different distibutions

Expect LB shuffle because of
- Randomness
- Little amount of data
- Different public/private distributions


# V. Metrics optimization

## 1) Regression metrics
- **MSE**: Mean Squared Error. Best constant: target mean 
- **RMSE**: Root Mean Squared Error. Best constant: target mean. More intuitive than MSE. It is not completely interchangeable for gradient-based methods as their gradients are not equal (there is a scaling factor)
- **R-squared**: How much our model is better than the constant baseline. Best constant: target mean. MSE, RMSE, R-squared are the same from an optimization perspective
- **MAE**: Mean absolute error. Best constant: target median. Use it when you have big outliers in the data that you don't want to penalize much. Problem: The gradient is not defined when the prediction is perfect.
- **(R)MSPE**: Mean Squared Percentage Error. Weighted version of MSE so that it cares about relative errors. Best constant: weighted target meann (objects with a small target value have higher weights)
- **MAPE**: Mean Absolute Percentage Error. Weighted version of MAE so that it cares about relative errors. Best constant: weighted target median (objects with a small target value have higher weights)
- **(R)MSLE**: Root Mean Square Logarithmic Error. Used in the same situation than MSPE and MAPE as it cares about relative errors. However it penalizes more lower predictions than higher ones.


## 2) Classification metrics :
- **Accuracy**: How frequently our class prediction is correct. Best constant: predict the most frequent class
- **LogLoss**: Logloss strongly penalizes completely wrong answers (remember that it looks at posterior probabilities). For example if we have a classifier assigning 0.4 score to an object of class 1, and other assigning 0.1, the later will be more penalized on this particular object. Best constant: set $a_i$ to frequency of $i-th$ class. TODO: when to use it?
- **AUC-ROC**: Depends only on ordering of the predictions, not on absolute values. Can be seen as a pairwise loss: probability of pair objects to be ordered in the right way. Best constant: All constants give same score. Random predictions lead to AUC = 0.5
- **Cohen’s (Quadratic weighted) Kappa**: Uses a baseline accuracy (i.e. percentage of largest class) to normalize accuracy. Very similar to what R2 does with MSE. There are versions of the metric using weights.

# VI. Metrics optimization

**General approaches for metrics optimization**

Target metric is what we want to optimize; however, optimization loss is what model optimizes

Just run the right model!
- MSE, Logloss
• Preprocess train and optimize another metric
- MSPE, MAPE, RMSLE, ...
• Optimize another metric, postprocess predictions
- Accuracy, Kappa
• Write custom loss function
- Any, if you can

Optimize another metric, use early stopping
- Optimize metric M1, monitor metric M2. Stop when M2 score is the best

## Regression metrics optimization

**MSE**: 
Can be found in many different libraries. Other names: L2 loss

**MAE**: 
Available only in some libraries. Similar to huber loss (Mix between MSE and MAE). Other names for MAE: L1, quantile loss

**MSPE and MAPE**:
Not commonly implemented. Options:
- Use the fact that they are weighted versions of MSE and MAE and tune the parameter 'sample_weights' in our models when available. Use MSE/MAE to optimize. 
- Re-sample the training set with the corresponding weights `df.sample(weights=sample_weights)` (See slides corresponding to this topic). Test set stays as it is. Use MSE/MAE to optimize. Usually one needs to resample many times and average the result.

**RMSLE**:
1) In the training set transform the target: $z_i=log(y_i+1)$. 
2) Fit a model with MSE loss
3) Transform the predictions back: $y_{pred}=exp(z_{pred_i}-1)$

## Classification metrics optimization

**LogLoss**:
Can be found in many different libraries. Note that it requires model to output posterior probabilities (i.e. if we take all objects with score 0.8, 80% of them will have class 1 and 20% of them will have class 0). If not it's predictions should be calibrated using methods such as:
- Platt scaling: Just fit Logistic Regression to your predictions (like in stacking)
- Isotonic regression: Just fit Isotonic Regression to your predictions
(like in stacking)
- Stacking: Just fit XGBoost or neural net to your prediction and use Logloss 

**Accuracy**:
There is no easy way to optimize it (gradient is either 0 or inf). Recommendation: Fit any metric and tune treshold!

**AUC**:
Although the loss function of AUC has zero gradients almost everywhere, exactly as accuracy loss, there exists an algorithm to optimize AUC with gradient-based methods, and some models implement this algorithm

**(Quadratic weighted) Kappa**:
Optimize MSE (with soft predictions) and tune thresholds

# VII. Advanced feature engineering

## Mean encoding
Use them for categorical variables with a lot of different values.

* Simplest case:  Encode each level of the categorical variable with the corresponding target mean:
```
means = x_train.groubpy(col).target.mean()
train_new[col+'_mean_target'] = train_new[col].map(means)
val_new[col+'_mean_target'] = val_new[col].map(means)
```
* Alternatives: $ln(Goods/Bads)*100$ (weight of evidence); count number of ones: $sum(target)$; $Goods - Bads$

Why does it work? It gives some logical order when there are too many categories.
When is it useful?  For algorithms like GBDT: one of the few downsides is its inability to handle high cardinality categorical variables, because trees have limited depth; with mean encodings, we can compensate for this limitation. One sign that may indicate the need for this encoding is to get better performance (without overfitting) for larger tree depths. This indicates trees need a huge number of splits to extract information from some variables


### Validation/Regulatization

It's got to be impeccable in order to avoid having leakages from the target variable. It needs regulatization. 

1) CV loop inside training data: For a given data point, we don't want to use target variable of that data point. So we separate the data into K subsets (folds). To get the mean encoding for a particular subset, we don't use data points from that subset and estimate the encoding only on the rest of subsets. Usually decent results with 4-5 folds. Fill nans with global mean (check implementation on slides).

2) Smoothing
Based on this idea: if a category has a lot of values, then we can trust the mean encoding, but if category is rare it's the opposite. 
$/frac{mean(target)*nrows + globalmean*alpha}{nrows+alpha}$ ???
It has hyper parameter alpha that controls the amount of regularization. When alpha is zero, we have no regularization, and when alpha approaches infinity everything turns into globalmean. In some sense alpha is equal to the category size we can trust.
We can combine it with, for example, CV loop regularization. 

3) Adding random noise: Meaning ecoding will have a better quality for the training data than for the test data. And by adding noise, we simply degrade the quality of encoding on training data. Pretty unstable and hard to make it work, so it will take some time.

4) Sorting and calculating expanding mean
We fix some sorting order of our data and use only rows from zero to $n-1$ to calculate encoding for row n (check implementation on slides). This method introduces the least amount of leakage from target variable and it requires no hyper parameter tuning. The only downside is that feature quality is not uniform. To overcome this, we can average models on encodings calculated from different data permutations. Built-in in CatBoost (good for categorical features).


### How to do mean encodings for other tasks?

* Regression: Use percentiles, standard dev., distribution bins
* Multiclass classification: For every feature we want to encode we will have n encodings, where n is the number of classes to predict
* Many-to-many relations: Create a "mean encoding vector" for every entity and then generate statistics on that vector
* Timeseries: Rolling statistics of target variable
* Interactions and numerical features: Analize fitted model. Selecting interactions and binning numeric results.


### Summary
Main advantages:
* Compact transformation of categorical variables
* Powerful basis for feature engineering
Disadvantages:
* Need careful validation, there a lot of ways to overfit
* Significant improvements only on specific datasets






#General advice

* Plot histograms of variables. Check that a feature looks similar between train and test set
* Plot features versus the target variable and vs time (bin numerical features)
* Stratified validation: Create a validation approach that resembles the testing set (or population data in a real-life case)
* Is time important? Split by time and do time-based validation
* Use early stopping
* Stay with a quick model such as lightGBM
* Only when satisfied with feature engineering use ensembles


Copy feature engineering:


Modeling:

Ensembling:
