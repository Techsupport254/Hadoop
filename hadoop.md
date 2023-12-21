### Step 1: Ingest the Data into Hadoop DFS Data Lake


```bash
hdfs dfs -put "C:/Users/kirui/Desktop/Freelance/Hadoop/owid-covid-data.csv" /path/in/hdfs
```

### Step 2: Extract Data Using PySpark
Once the data is in HDFS, extract it using PySpark:

```python
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("Covid19DataAnalysis").getOrCreate()

# Load the dataset from HDFS
df = spark.read.csv("hdfs:///path/in/hdfs/owid-covid-data.csv", header=True, inferSchema=True)
```

### Step 3: Pre-process the Extracted Data
Data pre-processing can involve several steps:

```python
from pyspark.sql.functions import to_date

# Convert date column to date format
df = df.withColumn('date', to_date(df['date'], 'yyyy-MM-dd'))

# Handling missing values
df = df.na.fill(0)  # or df.na.drop()

# Filter for specific data, e.g., Kenya
kenya_df = df.filter(df['location'] == 'Kenya')
```

### Step 4: Predictive Analytics
Choose a model based on the target (deaths, confirmed cases, recovery rates):

```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Define features and label
assembler = VectorAssembler(inputCols=['total_cases', 'new_cases', 'population'], outputCol="features")
data = assembler.transform(kenya_df).select('features', 'total_deaths')

# Split data
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Define and train the model
lr = LinearRegression(featuresCol='features', labelCol='total_deaths')
lr_model = lr.fit(train_data)
```

### Step 5: Visualize the Model
Visualization typically requires converting data to a Pandas DataFrame:

```python
import matplotlib.pyplot as plt

# Make predictions
predictions = lr_model.transform(test_data)

# Convert to Pandas DataFrame for visualization
pandas_df = predictions.select("total_deaths", "prediction").toPandas()

# Visualization
plt.scatter(pandas_df['total_deaths'], pandas_df['prediction'])
plt.xlabel('Actual Total Deaths')
plt.ylabel('Predicted Total Deaths')
plt.title('Actual vs Predicted Total Deaths')
plt.show()
```

### Step 6: Test the Model

```python
from pyspark.ml.evaluation import RegressionEvaluator

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="total_deaths", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {rmse}")
```
![Alt text](image.png)