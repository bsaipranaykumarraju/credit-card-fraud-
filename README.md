import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import plotly.express as px
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

import umap
import umap.plot
from pycaret.classification import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay

import scipy.stats as stats

card_df = pd.read_csv('../input/creditcard-fraud-detection/creditcard.csv')
card_org = card_df.copy()
colors = ['gold', 'mediumturquoise']
labels = ['Normal','Fraud']
values = card_df['Class'].value_counts()/card_df['Class'].shape[0]
fig = go.Figure(data=[go.Pie(labels = labels,
                             values=values,hole=.3)])
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='white', width=0.1)))
fig.update_layout(
    title_text="Credit Card Fraud",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
)
fig.show()
card_df.head(5).T.style.set_properties(**{'background-color': 'black',
                           'color': 'white',
                           'border-color': 'white'}))
card_df.info()
card_df.describe().style.set_properties(**{'background-color': 'black',
                           'color': 'white',
                           'border-color': 'white'})
import missingno as msno
msno.matrix(card_df)sns.set(style="ticks", context="talk",font_scale = 1)
plt.style.use("dark_background")
plt.figure(figsize = (8,6))
ax = card_df.dtypes.value_counts().plot(kind='bar',grid = False,fontsize=20,color='grey')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+ p.get_width() / 2., height + 0.2, height, ha = 'center', size = 25)
sns.despine()
card_df = card_df.sample(n = 20000,random_state=42)
card_df['Amount'].min()
numerical_cols = [cname for cname in card_df.loc[:,:'Amount'].columns if 
                  card_df[cname].min() >= 0 and
                  card_df[cname].dtype in ['int64','float64']]
from scipy.stats import skew
plt.figure(figsize=(8, 8))
skew_features = card_df[numerical_cols].apply(lambda x : skew(x))
skew_features = skew_features[skew_features > 0.5].sort_values(ascending=False)
ax = sns.barplot( x =skew_features.index,y=skew_features.values,color='grey')
for p in ax.patches:
    height = p.get_height().round(1)
    ax.text(p.get_x()+ p.get_width()/2.5, height-4, height, ha = 'left', size = 50)
plt.xticks(rotation=45)
plt.text(0.01,1.2, 'Threshold',color='red')
plt.axhline(y=1, color='green', linestyle='--', linewidth=3)
plt.title('Skewness',fontsize=30)
sns.despine()
def check_normality(feature):
    plt.figure(figsize = (8,8))
    ax1 = plt.subplot(1,1,1)
    stats.probplot(card_df[feature],dist = stats.norm, plot = ax1)
    ax1.set_title(f'{feature} Q-Q plot',fontsize=20)
    sns.despine()

    mean = card_df['Amount'].mean()
    std = card_df['Amount'].std()
    skew = card_df['Amount'].skew()
    print(f'{feature} : mean: {0:.4f}, std: {1:.4f}, skew: {2:.4f}'.format(mean, std, skew))
		def plot_hist(feature):
    fig = px.histogram(card_df, x=feature, 
                       color="Class",
                       marginal="box",
                       barmode ="overlay",
                       histnorm ='density'
                      )  
    fig.update_layout(
        title_text=f"{feature} Distribution",
        title_font_color="white",
        legend_title_font_color="yellow",
        paper_bgcolor="black",
        plot_bgcolor='black',
        font_color="white",
    )
    fig.show()
		from scipy import stats
plot_hist("Amount")
check_normality("Amount")
card_df[skew_features.index] = np.log1p(card_df[skew_features.index])
plot_hist("Amount")
check_normality("Amount")
import plotly.express as px
plt.figure(figsize=(20,20))
corr=card_df.corr().round(1)
fig = px.imshow(corr)
fig.update_layout(
    title_text="Credit Card Fraud",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
)
fig.show()
features = [
    "V3",
    "V7",
    "V10",
    "V11",
    "V12",
    "V14",
    "V16",
    "V17",
]

plt.figure(figsize=(13,8))
ax = abs(card_df[features].corrwith(card_df.Class)).sort_values(ascending=False).plot(kind='bar',color='grey',fontsize=20)
for p in ax.patches:
    height = p.get_height().round(2)
    ax.text(p.get_x() + p.get_width() / 2., height+0.007, height, ha = 'center', size = 30)
sns.despine()
def get_preprocessed_df(df=None):
    df_copy = df.copy()
    df_copy.drop('Time', axis=1, inplace=True)
    return df_copy

def get_train_test_dataset(df=None):
    df_copy = get_preprocessed_df(df)
    X_features = df_copy.loc[:,:'Amount']
    y_target = df_copy.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X_features,
                                                        y_target, 
                                                        test_size=0.3, 
                                                        random_state=0, 
                                                        stratify=y_target)
    return X_train, X_test, y_train, y_test

	X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)
 fig = px.histogram(card_df, x="V14", 
                   color="Class",
                   marginal="box",
                   barmode ="overlay",
                   histnorm ='density'
                  )  
fig.update_layout(
    title_text="Orignal Distribution",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
)
fig.show()
colors = ['gold', 'mediumturquoise']
fig = px.scatter(card_df, x="V17", y="V14", color="Class")
fig.update_layout(
    title_text="Original Scatter Plot",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
		)
	card_df.loc[:,:'Amount'].info()
 import seaborn as sns
import matplotlib.pyplot as plt
import umap
import pandas as pd
import numpy as np
sampled_card_df = card_df.sample(frac=0.1, random_state=42)
mapper = umap.UMAP().fit(sampled_card_df.loc[:,:'Amount'])
embedding = mapper.transform(sampled_card_df.loc[:,:'Amount'])
sns.set(style="ticks", context="talk", font_scale=1)
plt.style.use("dark_background")
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=sampled_card_df['Class'], cmap='coolwarm', s=10)
plt.title("UMAP Projection of Credit Card Dataset", fontsize=20)
plt.colorbar(label='Class')
plt.xlabel("UMAP Feature 1")
plt.ylabel("UMAP Feature 2")
plt.show()
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print('Feature/label dataset for training before applying SMOTE: ', X_train.shape, y_train.shape)
print('Feature/label dataset for training after applying SMOTE: ', X_train_smote.shape, y_train_smote.shape)
print('Distribution of label values after applying SMOTE:\n',pd.Series(y_train_smote).value_counts())
card_df_smote = pd.concat([X_train_smote,y_train_smote],axis=1)
fig = px.histogram(card_df_smote, x="V14", 
                   color="Class", 
                   marginal="box",
                   barmode ="overlay",
                   histnorm ='density'
                  )  
fig.update_layout(
    title_text="Oversampled by SMOTE",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
)
fig.show()
fig = px.scatter(card_df_smote, x="V17", y="V14", color="Class")
fig.update_layout(
    title_text="Oversampled by SMOTE",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white"
)
sns.set(style="ticks", context="talk",font_scale = 1)
plt.style.use("dark_background")

mapper = umap.UMAP().fit(card_df_smote.loc[:,:'Amount']) 
umap.plot.points(mapper, labels=card_df_smote.loc[:,'Class'], theme='fire')
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
print('Feature/label dataset for training before applying RandomOverSampler: ', X_train.shape, y_train.shape)
print('Feature/label dataset for training after applying RandomOverSampler: ', X_train_ros.shape, y_train_ros.shape)
print('Distribution of label values after applying RandomOverSampler: \n', pd.Series(y_train_ros).value_counts())
card_df_ros = pd.concat([X_train_ros,y_train_ros],axis=1)
fig = px.histogram(card_df_ros, x="V14", 
                   color="Class", 
                   marginal="box",
                   barmode ="overlay",
                   histnorm ='density'
                  )  
fig.update_layout(
    title_text="Oversampled by RandomOverSampler",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
)
fig.show()
fig = px.scatter(card_df_ros, x="V17", y="V14", color="Class")
fig.update_layout(
    title_text="Oversampled by SMOTE",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
)
mapper = umap.UMAP().fit(card_df_ros.loc[:,:'Amount']) 
umap.plot.points(mapper, labels=card_df_ros.loc[:,'Class'], theme='fire')
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=0)

X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
print('Feature/label dataset for training before applying ADASYN: ', X_train.shape, y_train.shape)
print('Feature/label dataset for training after applying ADASYN: ', X_train_adasyn.shape, y_train_adasyn.shape)
print('Distribution of label values after applying ADASYN: \n', pd.Series(y_train_adasyn).value_counts())
card_df_adasyn = pd.concat([X_train_adasyn,y_train_adasyn],axis=1)
fig = px.histogram(card_df_adasyn, x="V14", 
                   color="Class", 
                   marginal="box",
                   barmode ="overlay",
                   histnorm ='density'
                  )  
fig.update_layout(
    title_text="Oversampled by ADASYN",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
)
fig.show()
fig = px.scatter(card_df_adasyn, x="V17", y="V14", color="Class")
fig.update_layout(
    title_text="Oversampled by ADASYN",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
		)
	mapper = umap.UMAP().fit(card_df_adasyn.loc[:,:'Amount']) 
umap.plot.points(mapper, labels=card_df_adasyn.loc[:,'Class'], theme='fire')
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)

X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
print('Feature/label dataset for training before applying RandomUnderSampler: ', X_train.shape, y_train.shape)
print('Feature/label dataset for training after applying RandomUnderSampler: ', X_train_rus.shape, y_train_rus.shape)
print('Distribution of label values ​​after applying RandomUnderSampler: \n', pd.Series(y_train_rus).value_counts())
card_df_rus = pd.concat([X_train_rus,y_train_rus],axis=1)
fig = px.histogram(card_df_rus, x="V14", 
                   color="Class", 
                   marginal="box",
                   barmode ="overlay",
                   histnorm ='density'
                  )  
fig.update_layout(
    title_text="Undersampled by RandomUnderSampler",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
)
fig.show()
fig = px.scatter(card_df_rus, x="V17", y="V14", color="Class")
fig.update_layout(
    title_text="Undersampled by RandomUnderSampler",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
)
mapper = umap.UMAP().fit(card_df_rus.loc[:,:'Amount']) 
umap.plot.points(mapper, labels=card_df_rus.loc[:,'Class'], theme='fire')
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import InstanceHardnessThreshold
iht = InstanceHardnessThreshold(random_state=0,
                               estimator=LogisticRegression(solver='lbfgs', multi_class='auto'))

X_train_iht, y_train_iht = iht.fit_resample(X_train.loc[:,features], y_train)
print('Feature/label dataset for training before applying InstanceHardnessThreshold: ', X_train.shape, y_train.shape)
print('Feature/label dataset for training after applying InstanceHardnessThreshold: ', X_train_iht.shape, y_train_iht.shape)
print('Distribution of label values after applying InstanceHardnessThreshold: \n', pd.Series(y_train_iht).value_counts())
card_df_iht = pd.concat([X_train_iht,y_train_iht],axis=1)
fig = px.histogram(card_df_iht, x="V14", 
                   color="Class", 
                   marginal="box",
                   barmode ="overlay",
                   histnorm ='density'
                  )  
fig.update_layout(
    title_text="Undersampled by Instance Hardness Threshold",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
)
fig.show()
fig = px.scatter(card_df_iht, x="V17", y="V14", color="Class")
fig.update_layout(
    title_text="Undersampled by RandomUnderSampler",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
)
mapper = umap.UMAP().fit(card_df_iht.loc[:,:'V17']) 
umap.plot.points(mapper, labels=card_df_iht.loc[:,'Class'], theme='fire')
from imblearn.under_sampling import NearMiss
nm = NearMiss()

X_train_nm, y_train_nm = nm.fit_resample(X_train, y_train)
print('Feature/label dataset for training before applying RandomUnderSampler: ', X_train.shape, y_train.shape)
print('Feature/label dataset for training after applying RandomUnderSampler: ', X_train_nm.shape, y_train_nm.shape)
print('Distribution of label values after applying RandomUnderSampler: \n', pd.Series(y_train_nm).value_counts())
card_df_nm = pd.concat([X_train_nm,y_train_nm],axis=1)
fig = px.histogram(card_df_nm, x="V14", 
                   color="Class", 
                   marginal="box",
                   barmode ="overlay",
                   histnorm ='density'
                  )  
fig.update_layout(
    title_text="Undersampled by Near Miss",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
)
fig.show()
fig = px.scatter(card_df_nm, x="V17", y="V14", color="Class")
fig.update_layout(
    title_text="Undersampled by NearMiss",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
)
mapper = umap.UMAP().fit(card_df_nm.loc[:,:'Amount']) 
umap.plot.points(mapper, labels=card_df_nm.loc[:,'Class'], theme='fire')
train_df =pd.concat([X_train_smote,y_train_smote],axis=1)
X_test_smote, y_test_smote = smote.fit_resample(X_test, y_test)
colors = ['gold', 'mediumturquoise']
labels = ['Normal','Fraud']
values = train_df['Class'].value_counts()/train_df['Class'].shape[0]
fig = go.Figure(data=[go.Pie(labels = labels,
                             values=values,hole=.3)])
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(
    title_text="Credit Card Fraud",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
)
fig.show()
from pycaret.classification import setup

# Assuming train_df is your DataFrame containing your training data
classifier = setup(data=train_df, preprocess=False, target='Class', silent=True)
dt = create_model('dt')
catboost = create_model('catboost')
import pandas as pd
from pycaret.classification import *

# Load your dataset
df = pd.read_csv(r'credit card.csv')

# Assuming you have loaded and preprocessed your data, then:

# Set up PyCaret
exp_clf = setup(data=df, target='target_column')

# Create decision tree model
dt = create_model('dt')

# Create CatBoost model
catboost = create_model('catboost')

# Tune the decision tree model
tuned_dt = tune_model(dt, optimize='AUC')

# Tune the CatBoost model
tuned_catboost = tune_model(catboost, optimize='AUC')
import pandas as pd
import matplotlib.pyplot as plt  # Import matplotlib
from pycaret.classification import *

# Assuming you have loaded and preprocessed your data
# df = pd.read_csv('your_dataset.csv')

# Set up PyCaret
exp_clf = setup(data=df, target='target_column')

# Create and tune a decision tree model
tuned_dt = tune_model(create_model('dt'), optimize='AUC')

# Use matplotlib to set the figure facecolor to grey
with plt.rc_context({'figure.facecolor':'grey'}):
    interpret_model(tuned_dt)
		# Set up PyCaret
exp_clf = setup(data=df, target='target_column')

# Create and tune a CatBoost model
tuned_catboost = tune_model(create_model('catboost'), optimize='AUC')

# Use matplotlib to set the figure facecolor to grey
with plt.rc_context({'figure.facecolor':'grey'}):
    interpret_model(tuned_catboost)
# Set up PyCaret
exp_clf = setup(data=df, target='target_column')

# Create and tune a CatBoost model
tuned_catboost = tune_model(create_model('catboost'), optimize='AUC')

# Use matplotlib to set the figure facecolor to grey
with plt.rc_context({'figure.facecolor':'grey'}):
    interpret_model(tuned_catboost)
# Set up PyCaret
exp_clf = setup(data=df, target='target_column')

# Create and tune a CatBoost model
tuned_catboost = tune_model(create_model('catboost'), optimize='AUC')

# Use matplotlib to set the figure facecolor to grey
with plt.rc_context({'figure.facecolor':'grey'}):
    interpret_model(tuned_catboost)# Set up PyCaret
exp_clf = setup(data=df, target='target_column')

# Create and tune a CatBoost model
tuned_catboost = tune_model(create_model('catboost'), optimize='AUC')

plot_model(tuned_dt, plot='feature')
plot_model(tuned_catboost, plot='feature')
plt.figure(figsize=(8, 8))
plot_model(tuned_catboost, plot='boundary')
plt.figure(figsize=(8, 8))
plot_model(tuned_dt, plot='boundary')
plt.figure(figsize=(8, 8))
with plt.rc_context({'figure.facecolor':'black','text.color':'black'}):
    plot_model(tuned_dt, plot='tree')
plt.figure(figsize=(8, 8))
plot_model(tuned_dt, plot='learning')
plt.figure(figsize=(8, 8))
plot_model(tuned_dt, plot='class_report')
final_model_result = confusion_matrix(y_test_smote, pred)
accuracy = accuracy_score(y_test_smote , pred)
precision = precision_score(y_test_smote , pred)
recall = recall_score(y_test_smote , pred)
f1 = f1_score(y_test_smote,pred) 
print('accuracy: {0:.4f}, precision: {1:.4f}, recall: {2:.4f},\
F1: {3:.4f}'.format(accuracy, precision, recall, f1))
plt.figure(figsize=(8, 6))
ax = sns.heatmap(final_model_result, cmap = 'YlGnBu',annot = True, fmt='d')
ax.set_title('Confusion Matrix (final_model)')
