# Arvato customer Segmentation

This is my Data Scientist Capstone project. The goal of this project is to analyze demographics information  and identify clusters of the general population, customer population and use this information for initiating marketing campaigns to potential customers.Also not all leads respond, based on historical response, try to identify the leads whose response probability is higher for targeted campaigns. 

## Summary
The project consists of three parts 
1. Unsupervised Learning for clustering the general population of Germany and also map customers of a mail order company to the appropriate clusters.
2. Supervised Learning for predicting mail campaign response.
3. Submitting prediction response to Kaggle Data Science competition.

## Medium Blog Post
[Click here for more details](https://aravind-deva.medium.com/data-scientist-capstone-project-real-life-customer-segmentation-4d1441e01855)

## Project Structure(project-disaster)
```

├───Project-Capstone
│   └───Arvato Project Workbook.ipynb
│   ├───Arvato Project Workbook.html

```

### Instructions:
1. *Libraries:*
    This project requires libraries such as numpy,pandas,sklearn,imblearn,xgboost,scipy for EDA and model development. This project uses visualization libraries such as matplotlib and seaborn
	
2. *Installation:*
	
	!pip install -U numpy
	!pip install imbalanced-learn
	!pip install --upgrade setuptools
	!pip install --upgrade pip
	!pip install xgboost
	
3. *Execute the notebook*
	You can run the entire notebook at once by clicking on the 'Restart Kernel and then rerun the whole notebook 

4. *Local vs Cloud:* 
	This project is data-heavy.It is preferred to run this notebook on a cloud platform such as AWS,GCP,Azure

### Data Insights

1. The dataset consists of Demographics data for the general population of Germany 891 211 persons (rows) x 366 features (columns).Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
2. For feature selection, I have found the %change in entropy of each variable to consider it for further analysis. With 10% cut off I have got 117 columns after removing unnecessary columns.
3. Since **0,-1,9** are the general unknown values representation.I have imputed null values with careful study of the columns and their unknown column representation 
4. I have done PCA and 100 columns were able to explain 99% of the variance 
5. Simple 2component PCA plot suggests data is not clearly separated.

![PCA Variance](https://github.com/aravind-deva/Data-Science/blob/main/Project-Capstone/pca.png)

## ML 
1. I have used KMeans algorithm as DBScan needs separate clusters and Meanshift is computally intensive
2. I have transformed Customer Dataset using PCA above and mapped them to clusters , Most of the customers are concentrated on clusters 11,6,3 
3. ![Clusters](https://github.com/aravind-deva/Data-Science/blob/main/Project-Capstone/Customer%20Segmentation.png)
4. The mail order campaign training data was targeted among the same clusters are above along with RESPONSE column
5. The dataset is imbalanced. So the accuracy metric is favored towards the majority class.The 
6. The best way is to make the classes balanced by using **oversampling/undersampling/SMOTE and other** techniques. My personal was SMOTE (It preserves all information and generates/augments new data points with in the same cluster of data points using an average sample strategy)
7. That said,SMOTE was slighly worse compared to models trained with out SMOTE. Thats because of overfitting/not using advanced techniques for oversampling. 

## Kaggle
I have achieved an ROC- AUC score of 0.79 with only 109 columns and GradientBoostClassifier. This is my current leader board position(133) in the data Science Kaggle competition as on 26th Jun 2021
![Kaggle Leaderboard](https://github.com/aravind-deva/Data-Science/blob/main/Project-Capstone/Rank.PNG)

# License
Licensed under [LICENSE](https://github.com/aravind-deva/Data-Science/blob/main/LICENSE)

# Acknowledgements

Special thanks to Udacity for this course and Bertelsmann/Arvato for providing us real life datasets 
