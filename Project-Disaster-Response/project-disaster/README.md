# Disaster Response Pipeline Project

### Instructions:Local and on Workspace/
1. Local PC Only:
    - conda create -n deploy_env python==3.6.13 (Creates a virtual environment with Heroku supported python version )
    - conda activate deploy_env (activate the environment)
    - pip install -r requirements.txt (installs the required dependencies)

2. Local PC & Workspace:
   Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the project home directory to run your web app.
    `python app/run.py`

4. Local : 
      Go to http://localhost:3001/
5. Workspace: 
    Follow the instructions in udacity:
    
   /home/workspace# env | grep WORK
    WORKSPACEDOMAIN=udacity-student-workspaces.com
    WORKSPACEID=
### Cloud Deployment
6. The project has been deployed to Heroku cloud app hosting platform (https://aravind-deva-dsnd-disaster.herokuapp.com/)
- Requires Procfile and requirements.txt
### Web App details
    It has home page with two dashboards (pie chart and bar chart ) to indicate the top classified messages 
    It has a text box to identify the category using a machine learning model
    The classified page also links to various organizations to reach for the relevant categories.

### Data Insights
1. The dataset consists of disaster event messages and the corresponding hand labelled classification of the message into various categories such as earthquake,flood,storm related. A single message can be categorized as multi categories. So this is a **multi label** classification.
2. However the classification data set needs to be transformed for analysis.
   "related-1;request-0;offer-0;aid_related-0;medical_help-0;...."
3. I used dataframe str.split(';',expand=True) to split and convert each category to a column. I then applied a lambda function to trim off characters and make it as a numeric column
4. I removed columns which don't have atleast 2 unique values for a column(Ex: *child_alone*) since there is nothing to learn from a invariant column. I replaced column values with more than 2 values with the column's mode (*related*) . Some ML algorithms/metrics require atleast 2 labels. 
5. I merged the messages dataframe and the cleansified hand-labelled data frame and saved it in file database

## ML 
1. The message(text) is tokenized , lemmatized (verb and nouns) using nltk to convert it into a TF-IDF Vectorized form
2. I have use MultiOutputClassifier with a base AdaboostClassifier(with class_weight). I used GridSearchCV  cross validation to find the ideal n_estimators
3. I have also tried SVC(rbf & linear kernels) and KNeighborsClassifier. KNearest neighbors is a very simple yet a great classifier.However for this data set, Adaboost is winning over others by a decent margin.
4. The dataset is imbalanced. So the accuracy metric is favoured towards the highest class.The weighted-average f1-score is a decent metric to evaluate.
5. The best way is to make the classes balanced by using **oversampling/undersampling/SMOTE and other** techniques. 
6. That said, I chose to use a simple *class_weight='balanced'* in the base estimator. With this there is more penalty for misclassifying a minority labelled data point. Thus I have a achieved a non-zero f1-score for minority labelled datasets.
