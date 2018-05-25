import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns


file = 'D:/kaggle/ks-projects-201801.csv'
df = pd.read_csv(file, encoding='latin-1')

# Data Cleaning
df = df[df['goal'] > 100]   # Assuming that no sound project should be <$100
df = df[df['backers'] > 0]  # Assuming that no sound project should have backers == 0
df['year'] = pd.to_datetime(df['launched']).dt.year # Add year column
df['project_duration'] = pd.to_datetime(pd.to_datetime(df['deadline']) - \
                pd.to_datetime(df['launched'])).dt.day # Determine each project duration

df = df[~df['year'].isin(['1970', '2018'])] # Exclude 1970 and 2018
df['avg_backer_invest'] = df['pledged']/df['backers']


# Overall projects split
def plot_overview():
    all_cat = df['main_category'].value_counts()
    all_cat.plot.pie()
    plt.show()


# Top 10 categories analysis of success and failure
def plot_top10categoris():
    subdf = df[['main_category', 'state']]
    pivot = subdf.pivot_table(index='main_category', columns='state', aggfunc=len)
    pivot = pivot.sort_values(by='successful', ascending=False) [:10] # Top 10 categories
    attrs = pivot.index.tolist()
    attr_vals = np.array(pivot.values.tolist()).T

    N = len(attrs)
    ind = np.arange(N)
    width = 0.7
    p1 = plt.bar(ind, attr_vals[0], width)
    p2 = plt.bar(ind, attr_vals[1], width, bottom=attr_vals[0])
    p3 = plt.bar(ind, attr_vals[2], width, bottom=attr_vals[0]+attr_vals[1])
    p4 = plt.bar(ind, attr_vals[3], width, bottom=attr_vals[0]+attr_vals[1]+attr_vals[2])
    p5 = plt.bar(ind, attr_vals[4], width, bottom=attr_vals[0]+attr_vals[1]+attr_vals[2]+attr_vals[3])

    plt.legend([p1[0], p2[0], p3[0], p4[0], p5[0]], ['canceled', 'failed', 'live', 'successful', 'suspended', 'undefined'])
    plt.xticks(ind, attrs)
    plt.show()


# project counts by year
def plot_annual_project():
    annual_growth = df.year.value_counts().sort_index()
    index = annual_growth.index.tolist()
    vals = annual_growth.tolist()

    N = len(index)
    ind = np.arange(N)
    width = 0.7

    plt.bar(ind, vals, width=width)
    plt.xticks(ind, index)
    plt.show()


# Project state and Backers
def plot_backers_analysis():
    backersdf = df[['main_category', 'state', 'backers']]
    backers_pivot = backersdf.pivot_table(index='main_category', columns='state', aggfunc='mean')
    cols = np.array(backers_pivot.columns.tolist()).T[1]
    backers_pivot.plot.bar()
    plt.legend(cols)
    plt.xticks(rotation=45)
    plt.show()


# Applying classification models
def claasification():
    data = df[df['state'].isin(['successful', 'failed'])]
    X = data[['main_category', 'country', 'goal', 'pledged', 'backers',
               'project_duration', 'avg_backer_invest']] # Selecting features from data
    X = pd.get_dummies(X, columns=['main_category', 'country']) # Converting to dummies for modeling purposes
    X = X.as_matrix()
    y = data['state'].as_matrix()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Decision Tree classifier
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    # MLP classifier
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of Decision Tree: {}%'.format(accuracy * 100))


# Box plots for numerical data
def box_plot_analysis():
    box_df = df[['goal', 'pledged', 'backers', 'project_duration', 'avg_backer_invest']]
    box_df = box_df.apply(np.log, axis=1)
    box_df['state'] = df['state']
    sns.boxplot(data=box_df)
    plt.show()

    # Boxplot: goal vs state
    fig, ax = plt.subplots(figsize=(10,8))
    box_df.boxplot(column=['goal'], by='state', ax=ax)
    plt.show()


# Successful VS Unsuccessful projects by year
def plot_annual_success_by_year():
    state_year_df = df[['year', 'state']]
    state_year_df = state_year_df.pivot_table(index='year', columns='state', aggfunc=len)
    state_year_df['failed'] = state_year_df['canceled']+state_year_df['failed']+state_year_df['suspended']
    state_year_df = state_year_df[['successful', 'failed']]
    state_year_df.plot.bar()
    plt.show()


# Country VS Success of projects
def plot_projects_success_by_country():
    subdf_countries = df[['state', 'country']]
    subdf_countries = subdf_countries.pivot_table(index='country', columns='state', aggfunc=len)
    subdf_countries['failed'] = subdf_countries['canceled']+subdf_countries['failed']+subdf_countries['suspended']
    subdf_countries = subdf_countries[['successful', 'failed']].sort_values('successful')
    subdf_countries.plot.bar()
    plt.show()


# Scatter plot GOALS vs Pledged (LOG)
def plot_goal_pledged():
    goal_pledged_df = df[['goal', 'pledged']]
    goal_pledged_df = goal_pledged_df.apply(np.log, axis=1)
    goal_pledged_df.plot.scatter(x='goal', y='pledged', c='green')
    plt.show()


# Pledged VS Backers
def plot_pledged_backers():
    pledged_backers_df = df[['backers', 'pledged']]
    pledged_backers_df = pledged_backers_df.apply(np.log, axis=1)
    pledged_backers_df.plot.scatter(x='backers', y='pledged', c='green')
    plt.show()


# Highest pledging categories
def plot_highest_pledging_countries():
    category_pledged_df = df[['main_category', 'pledged']].pivot_table(index='main_category', aggfunc=sum)
    category_pledged_df = category_pledged_df.sort_values('pledged')
    category_pledged_df.plot.bar()
    plt.show()