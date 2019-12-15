# -*- coding: utf-8 -*-
"""
#  Assignment Done By Melih Kurtaran

The aim of this assignment is to get used dataframe manipulations with Pandas library.

"""

from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)

"""## Get to Know Your Data

#### Titanic Dataset

The dataset store the survival status of actual individual passengers
on the Titanic, which sank in 1912. Each row in the dataset corresponds to a passenger whose features are stored in columns. In this dataset, you can find detailed information for a passenger. For instance, the following list contain some of the attributes associated for each passenger in the dataset.

- name
- age
- gender
- number of siblings
- whether the passenger survived or not

In your first task, your goal is to read and store your dataset in a dataframe. And then, you will have some basic questions regarding the dataset.

#### Q1-Reading A File

Read _"titanic.tsv"_ file from your drive folder, store it as a dataframe named **titanic_df** and show the first 5 rows. 

p.s It is not a **comma-separated** file.
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd  # an alias for pandas
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

# My code

from os.path import join  # merging paths in a better way

data_path = "/content/gdrive/My Drive/"
filename = "titanic.tsv"

titanic_df = pd.read_csv(join(data_path, filename), sep='\t', header=0)

titanic_df.head(5)

"""#### Q2-Shape of your Dataset

Show the number of rows and columns stored in the dataframe **titanic_df**.
"""

# My code

r, c = titanic_df.shape

print("Number of rows:", r)
print("Number of columns:", c)

"""#### Q3-Column Names

Show the names of all columns in **titanic_df**.
"""

# My code

#1st method
print(titanic_df.columns.values)

print() 

#2nd method
titanic_df.columns

"""#### Q4-Data Types

Show the data type of each column.

```
PassengerId      int64
Survived         int64
Pclass           int64
Name            object
Sex             object
Age            float64
SibSp            int64
Parch            int64
Ticket          object
Fare           float64
Cabin           object
Embarked        object
dtype: object
```
"""

# My code

titanic_df.dtypes

"""#### Q5-Summarization

Show the descriptive statistics that summarize numeric columns.

|PassengerId|Survived|Pclass|Age|SibSp|Parch|Fare|
|---|---|---|---|---|---|---|
|count|891.000000|891.000000|891.000000|714.000000|891.000000|891.000000|891.000000|
|mean|446.000000|0.383838|2.308642|29.699118|0.523008|0.381594|32.204208|
|std|257.353842|0.486592|0.836071|14.526497|1.102743|0.806057|49.693429|
|min|1.000000|0.000000|1.000000|0.420000|0.000000|0.000000|0.000000|
|25%|223.500000|0.000000|2.000000|20.125000|0.000000|0.000000|7.910400|
|50%|446.000000|0.000000|3.000000|28.000000|0.000000|0.000000|14.454200|
|75%|668.500000|1.000000|3.000000|38.000000|1.000000|0.000000|31.000000|
|max|891.000000|1.000000|3.000000|80.000000|8.000000|6.000000|512.329200|
"""

# My code

titanic_df.describe()

"""## Dealing with NaN Values

Many real-world datasets may contain missing values for various reasons. They are often encoded as NaNs, blanks or any other placeholders. 

<p>
    <img src="https://media.geeksforgeeks.org/wp-content/uploads/isnull1.jpg"/>
    <br>
    <em>Fig. 1: A sample dataset with various NaN values.</em>
</p>


As you may expect, the titanic dataset contains NaN values as well. Now, you are asked to find and deal with NaN values.

#### Q1-Check NaN Columns

Show the columns in the dataset, which contains at least one NaN value.

**hint:** Series objects accept boolean arrays for selection.
"""

# My code

df_c = titanic_df[titanic_df.isna().any(axis=1)]

df_c

"""#### Q2-NaN values in a Particular Column

Show the number of NaN instances in _"Age"_ column.
"""

# My code

df_c.isnull().sum()["Age"]

"""#### Q3-NaN Percentage in Each Column

Show the percentage of NaN values in each column in descending order.

**hint:** You may use `sort_values` function on a series object.

```
Cabin          0.771044
Age            0.198653
Embarked       0.002245
Fare           0.000000
Ticket         0.000000
Parch          0.000000
SibSp          0.000000
Sex            0.000000
Name           0.000000
Pclass         0.000000
Survived       0.000000
PassengerId    0.000000
dtype: float64
```
"""

# My code

df_cc = df_c.isnull().sum() / r

df_cc.sort_values(ascending=False)

"""#### Q4- Removing the "Cabin" Column

Remove the "Cabin" column from your dataframe, and then show the top 5 rows of the resulting dataframe.
"""

# My code

df_a = titanic_df
df_a.drop("Cabin", axis=1, inplace=True)

df_a.head(5)

"""#### Q5-Removing Rows with NaN Instances

Remove the rows in which the "Embarked" column has an NaN value and then show that no NaN values exits in the column after the removing operation is done.

**hint:** You may check the parameters of `dropna` function.
"""

# My code

df_u = titanic_df.dropna(subset=['Embarked'])

df_u.isnull().sum()["Embarked"]

"""#### Q6-Filling NaN Values

At this point, the only column with NaN values should be the "Age" column.  

Fill the NaN values in the "Age" column with the mean age value of the passengers. Then, show that no NaN values exist in the dataframe.
"""

# My code
import numpy as np

dfs = df_u.fillna(value=titanic_df["Age"].mean())

dfs

"""## Extracting Insights

So far, you have read the dataset and dealt with NaN values.

Now, you'll extract some insights regarding the dataset, such as

- Survival Rate
- Gender Ratio among the passengers
- Frequency Tables

#### Q1-Survival Rate

Show the percentage of survived passengers.

_p.s._ Use 2 decimal points to print the results.
"""

# My code

def survivalRate(row):
  noOfPeople, noOfSurvived = 0, 0
  survival = row["Survived"]
  for i in survival:
    if i == 1:
      noOfSurvived += 1
    noOfPeople += 1
  return noOfSurvived / noOfPeople * 100

   

result = round(survivalRate(titanic_df),2)

print(result, "%")

"""#### Q2-Gender Distribution

Show the gender distribution among the passengers.

_p.s._ Use 2 decimal points to print the results.
"""

# My code


def genderDis(row):
  noOfPeople, noOfMale, noOfFemale = 0, 0, 0
  sex = row["Sex"]
  for i in sex:
    if i == "male":
      noOfMale += 1
    elif i == "female":
      noOfFemale += 1
    noOfPeople += 1   
  return noOfMale / noOfPeople * 100, noOfFemale / noOfPeople * 100

   

result = genderDis(titanic_df)

print("Male:",round(result[0],2), "%")
print("Female:",round(result[1],2), "%")

"""#### Q3-Mean Age of Survived Female Passengers

Show the mean age of survived female passengers.

**hint:** `"Survived"` column consists of binary values. `1` for survived and `0` for died passengers.
"""

# My code

withoutNullAge = titanic_df.dropna(subset=['Age'])
Survived = withoutNullAge[withoutNullAge.Survived == 1]
SurvivedFemale = Survived[Survived.Sex == "female"]

def meanAgeOfSurvivedFemalePassengers(row):
  sumOfAges, noOfSurvivedFemale = 0, 0
  Ages = row["Age"]
  for i in Ages:
      noOfSurvivedFemale += 1
      sumOfAges += i
  return sumOfAges / noOfSurvivedFemale

result = round(meanAgeOfSurvivedFemalePassengers(SurvivedFemale),2)

print("Mean age of survived female passengers:",result, "%")

"""#### Q4-Frequency Table

Show how the survival counts change with respect to `sex` and `pclass` (passenger class).

Create the table below, by a utilizing `group by` statement on your dataframe. And then, show the summarized `Survived` column.

**hint:** You may utilize `groupby` function with multiple labels.  
**hint2:** Watch out for the difference between single brackets (["a column"]) and double brackets ([["a column"]])
"""

# My code

titanic_df.groupby([titanic_df["Sex"],titanic_df["Pclass"],titanic_df["Survived"]])[["Survived"]].count()

"""#### Q5-Survival Probability

Calculate the conditional probability that a passenger survives given their gender is `female` and passenger class (`"Pclass"` column) is first class (1).

$P(Survived=1 \; | \; Sex=female, \, Pclass=1 )$

**hint:** You may use the frequency table above to extract count values with `multiindexing`. However, you may simply use a regular row selection as well.
"""

# My code

# P(A) / P(B)

A = titanic_df[(titanic_df["Survived"] == 1) & (titanic_df["Sex"] == "female") & (titanic_df["Pclass"] == 1)].shape[0]
B = titanic_df[(titanic_df["Sex"] == "female") & (titanic_df["Pclass"] == 1)].shape[0]

print(A/B)

"""## Data Manipulation

A substantial amount of time in any project will be spent on preparing, transforming and manipulating the data into the desired format.

In this section, you will start with reading a dataset of a different format which you have not seen before. Then, you'll create new columns and manipulate them.

#### Q1-Reading a Dataset with a Different Format

In the shared folder, you see two files named `adult.data` and `"adult.names"`. Although they are two separate files, they are just two parts of the same data source.  

**adult.data:** This file stores the actual data in a comma-separated format; however, the column names are absent. It's pretty much a regular csv file.

**adult.names:** This file stores the column names with a specific format. When you open the file, you'll see that it has lines starting with `"|"` character, which states that the line is just a comment. In addition, the file contains blank lines, consisting solely of new line character `"\n"`. And the actual column names start with an alphanumeric character appended with the data type of the column. Luckily, the column name and the data types are separated by a colon "`:`" character.

Your task is to read both of the files into a dataframe. A sample set of instructions can be found below.

1. Read *adult.data* as if it's a regular csv file into a dataframe. (Columns will be regular index values.)
  - Do not forget to set the `header` parameter to `None` to state there exists no column line in the file
2. Read the content of *adult.names* line by line.
3. If current line does not start with "|" or "\n", then it's a column name.
4. In order to extract the column name from the current line, split it by ":" and take the first element.
5. After you are done with all column names, assign them as the final column list of the dataframe.

At the end of this pipeline, you should obtan the dataframe below.
"""

# Commented out IPython magic to ensure Python compatibility.
# My code
import pandas as pd  # an alias for pandas
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

from os.path import join

data_path = "/content/gdrive/My Drive/"

filename = "adults.names"

cols = []

with open(join(data_path, filename)) as f:
  for line in f:
    
    if line[0] != "|" and line[0] != "\n":
      for word in line.split():
        i = word.find(':')
        if i != -1:
          cols.append(word[0:len(word)-1])
          
filename = "adults.data"

df_ex = pd.read_csv(join(data_path, filename), delimiter=",", header=None)

df_ex.columns = cols

df_ex.head(5)

"""#### Q2-Categorazing Salaries

The `"salary"` column stores continuous values. Your task is to categorize them based on a set of ranges. You may find the set of range and their corresponding categories below.

|Range|Category|
|---|---|
|1.000-2.999|low|
|3.000-5.999|med|
|6.000-10.000|high|

For instance, the firs 5 rows in the salary column should be converted into the following state.

|Original|Converted|
|---|---|
|3136|med|
|4784|med|
|8637|high|

After you are done with categorization, show the top 5 rows in the dataframe.
"""

# My code

def assignCategory(salary):
  if salary < 3000:
    return "low"
  elif salary < 6000:
    return "med"
  else:
    return "high"
  
df_ex["Category"] = df_ex["salary"].apply(assignCategory)  # resulting series will be stored in a new column

df_ex.head(5)

"""#### Q3-Creating a New Column

Create a new column named `"weekly_salary"` in the dataframe by combining `"salary"` and "`hours-per-week`" columns. Basically, the new column should store values obtained by diving values in `"salary"` column by "`hours-per-week`".

After the new column is created, again show the top 5 rows in the resulting dataframe.
"""

# My code

df_ex["weekly_salary"] = df_ex["salary"] / df_ex["hours-per-week"]

df_ex.head()

"""#### Q4-String Manipulation

In "`marital-status`" column, words are separated by dashes ("-"). Instead, convert each occurrence of a dash character with a white-space.

After the new column is created, again show the top 5 rows in the resulting dataframe.
"""

# My code

df_ex["marital-status"] = df_ex["marital-status"].str.replace("-"," ")

df_ex.head()

"""## Visualization

> Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.   
(https://www.tableau.com/learn/articles/data-visualization)

In the last set of tasks, you are required to visualize various phenomena from the `NBA` dataset.

The NBA dataset consist of seasonal statistics of each player from 1980 to 2017. Each row corresponds to a player with various attributes in a particular season. As a result, players may occur in multiple rows.

#### Q1-Reading NBA Dataset

Read and store the file named `"nba_players.csv"` into a dataframe, and show the top 5 rows.
"""

# My code

from os.path import join  # merging paths in a better way

data_path = "/content/gdrive/My Drive/"
filename = "nba_players.csv"

nba_df = pd.read_csv(join(data_path, filename), sep=',', header=0)

nba_df.head()

"""#### Q2-Convert the "Year" Column into DateTime Object

Initially, the "Year" column is of object (string) data type. Convert it to DateTime object. And show its data type after the conversion is done.
"""

# My code

nba_df['Year']= pd.to_datetime(nba_df['Year']) 

nba_df.dtypes

"""#### Q3-Overall Points in Time

Group your dataframe by the "year" column and find the total amount of points ("PTS" column) made in each season.  

Show how cumulative points change in each season with a line plot.

_p.s._ In 1998-1999 and 2011-2012 seasons, the league went through major lockouts. We expect to see striking downfall in overall points made in those seasons.
"""



# My code

nba_df['Year'] = nba_df['Year'].astype(str)
def get_year(datestr):
    return datestr.split('-')[0]

nba_df.set_index('Year',inplace=True)
tr_nba_df = nba_df["PTS"].transpose()

tr_nba_df
grouped_by_year = tr_nba_df.groupby(get_year).mean() * tr_nba_df.groupby(get_year).count()

plt.xlabel('Year')
plt.title('Overal Points Made between 1980-2017')
plt.xticks(range(0,38,5))
plt.plot(grouped_by_year, marker="o")

"""#### Q4-Top Scorers

Compute the top 5 scorers by accumulating the total points for each players. And show the result as a bar chart.
"""
# My code

grouped_by_player = nba_df.groupby("Player").mean() * nba_df.groupby("Player").count()

tobePlotted = grouped_by_player.nlargest(5, 'PTS')
plt.ylabel("Player")
plt.title("Top5 Scorers between 1980-2017")
bar_width=0.5
plt.barh(tobePlotted.index,tobePlotted["PTS"],bar_width)

"""#### Q5-Minutes Played vs. Personal Fouls

Show the relationship between minutes played ("MP" column) and personal fouls ("PF" column) as a scatter plot in which each dot represents a player.

**hint:** Setting the size of scatter points as 0.1 would yield a much better visual.
"""

# My code
plt.xlabel('MP')
plt.title('MP vs. PF')
plt.scatter(nba_df["MP"], nba_df["PF"], s=0.1)
