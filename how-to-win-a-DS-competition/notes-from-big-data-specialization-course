# Recap of main ML algorithms

* Linear models: Split space into 2 subspaces with a hyperplane (Ex.: Logistic Reg., SVM). Good for sparse, high dimensional data, where there are linear dependencies.
* Tree-based methods: Decision tree is the main construction block. Uses divide and conquer approach to recursively divide the space into subspaces (or boxes). In regression tasks aproximates points in the boxes with a constant. A bad choise when data has linear dependencies.
* kNN-based methods: Points close to each other are likely to have the same labels. Heavily rely on how to measure points "closeness"
* Neural Networks: Produce smooth separating curve (in contrast to decision trees).

No free lunch theorem: There is no method which outperforms all others for all tasks. There is no

# Feature processing

Preprocessing and generation pipelines depend on a model type.

For instance:
- Lineal models require hot-encoded variables if the dependencies between numerical features and target are not linear. 
- Decision-trees-based models do not need one hot encoding. These models also don't depend on feature scaling
- 

## Numeric features

### Preprocessing

#### Feature scaling:
We use preprocessing to scale all features to one scale, so that their initial impact on the model will be roughly similar.

Decision-trees-based models don't depend on feature scaling while Linear Models , KNN and neural networks do (regularization impact is proportional to feature scale and  gradient descent methods will go crazy without the proper scaling)

Types: 
- Min max scaling (distributions do not change)
- Standar scaler (Goal: mean==0 and std==1)


#### Outliers treatment
Outliers affect linear models. To prevent them we can use winzorisation: clip feature values ammong a low and a high percentile

#### Rank transformation
Set spaces between sorted feature values to be equal. Can be also used to treat outliers as with the rank they will be closer to other values. Linear models, KNN and NN can also benefit from this transformation. Can be applied to train+test data

#### Other transformations
These transformations drive too big values to the features average value. Besides, values closer to zero become more distinguishable 
- Log transform: `np.log(1+x)`
- Raising to the power <1: `np.sqrt(x+2/3)`

TIP: use ensemble methods with different preprocessings to achieve better results

### Features generation
Examples:
- Interactions: cLick rate, price per m2, calculated distances with Pytagoras theorem. In adition to Linear Models, these are beneficial for methods such as GBT as they experience dificultiexs with approximations of multiplications and divissions
- Fractional part (i.e. of prices)


## Categorical and ordinal features

## Datetime and coordinates

## Handling missing values



