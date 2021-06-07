# Disaster Response Pipeline Project

## Summary
The goal of this project is to analyze disaster data from Figure Eight containing real messages that were sent during disaster events and then use a ML pipeline to build a model that classifies disaster messages.
The project includes a web app where an emergency worker can input a new message and get classification results in several categories. It will also display visualizations of the data. 

## Project Structure(project-disaster)
```
app
| - templates
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- DisasterResponse.db # database to save clean data to
models
|- train_classifier.py
|- classifier.pkl # saved model
utils
|- token_fn.py # Utility function to tokenize texts while building the model and also while predicting the text
- requirements.txt #dependencies for Heroku deployment as well as local python env creation
- Procfile #Boot Up script on Heroku
- runtime.txt #Python version Stack selection by Heroku
README.md

```

### Instructions:
1. *Local PC:*
    - conda create -n deploy_env python==3.6.13 (Creates a virtual environment with Heroku supported python version )
    - conda activate deploy_env (activate the environment)
    - pip install -r requirements.txt (installs the required dependencies)

2. *Local PC & Workspace:*
   Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
        
        ![scores](https://github.com/aravind-deva/Data-Science/blob/main/Project-Disaster-Response/Scores.PNG)
3. *Run the following command in the **project home directory** to run your web app.*
    `python app/run.py`
   ### !! NOTE !! ## 
    - We have ensured installing joblib in the requirements.txt.However if in workspace,joblib is missing, then depending on the version of sklearn you might be running into an error with joblib.Please uncomment one or the other lines below in run.py
      
      ```
      #from sklearn.externals import joblib
      
      import joblib 
      ```

4. *Local:* 
      Go to http://localhost:3001/
5. *Workspace:*
    Follow the instructions in udacity:
    
   /home/workspace# env | grep WORK
    WORKSPACEDOMAIN=udacity-student-workspaces.com
    WORKSPACEID=
### Cloud Deployment
6. The project has been deployed to Heroku cloud app hosting platform (https://aravind-deva-dsnd-disaster.herokuapp.com/)
- Requires Procfile , requirements.txt ,runtime.txt
7.  Heroku may be booting/scaling up the app at various time windows.During that window, you may face timeout error while accessing the heroku webapp. Please retry it.
### Web App details
   - Feel free to browse (https://aravind-deva-dsnd-disaster.herokuapp.com/)
   - It has a home page with two dashboards (pie chart and bar chart ) to indicate the top classified messages 
   - It has a text box to identify the category using a machine learning model
   - The classified page also links to **various organizations** such as Red Cross,UNICEF,Ameri Cares etc., to reach out to.
   ![Classification Page](https://github.com/aravind-deva/Data-Science/blob/main/Project-Disaster-Response/Classfication%20Example.PNG)
   ![Home Page](https://github.com/aravind-deva/Data-Science/blob/main/Project-Disaster-Response/PieChart.PNG)
   ![Home Page](https://github.com/aravind-deva/Data-Science/blob/main/Project-Disaster-Response/Top%20Categories.PNG)

### Data Insights
1. The dataset consists of disaster event messages and the corresponding hand labelled classification of the message into various categories such as earthquake,flood,storm related. A single message can be categorized as multi categories. So this is a **multi label** classification.
2. However the classification data set needs to be transformed for analysis.
   "related-1;request-0;offer-0;aid_related-0;medical_help-0;...."
3. I used dataframe str.split(';',expand=True) to split and convert each category to a column. I then applied a lambda function to trim off characters and make it as a numeric column
4. I removed columns which don't have atleast 2 unique values for a column(Ex: *child_alone*) since there is nothing to learn from a invariant column. I replaced column values with more than 2 values with the column's mode (*related*) . Some ML algorithms/metrics require atleast 2 labels. 
5. I merged the messages dataframe and the cleansified hand-labelled data frame and saved it in file database

## ML 
1. The message(text) is tokenized , lemmatized (verb and nouns) using nltk to convert it into a TF-IDF Vectorized form
2. I have use MultiOutputClassifier with a base AdaboostClassifier(with class_weight). I used GridSearchCV  cross validation to find the ideal n_estimators ```
            AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,class_weight='balanced'),n_estimators=100) 
            ```
3. I have also tried SVC(rbf & linear kernels) and KNeighborsClassifier. KNearest neighbors is a very simple yet a great classifier.However for this data set, Adaboost is winning over others by a decent margin.
4. The dataset is imbalanced. So the accuracy metric is favored towards the majority class.The average precision/f1-score metric of individual class is highly effected by extreme values. Therefore the weighted-average f1-score/precision is a decent metric to asses.However the data set is imbalanced which leads to model's low skill levels in detecting true positives(if they are in minority)
5. The best way is to make the classes balanced by using **oversampling/undersampling/SMOTE and other** techniques. My personal choice would be to use SMOTE (It preserves all information and generates/augments new data points with in the same cluster of data points using an average sample strategy)
6. That said, I chose to use a simple *class_weight='balanced'* in the base estimator. With this there is more penalty for misclassifying a minority labelled data point. Thus I have a achieved a non-zero f1-score for minority labelled datasets.
