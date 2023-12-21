from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

# Initialize Spark Session
spark = SparkSession.builder.appName("Covid19DataAnalysis").getOrCreate()

# Load the dataset
df = spark.read.csv("owid-covid-data.csv", header=True, inferSchema=True)

# Convert date column to date format
df = df.withColumn('date', to_date(df['date'], 'yyyy-MM-dd'))

# Filter for Kenya
kenya_df = df.filter(df['location'] == 'Kenya')

# Fill missing values
kenya_df = kenya_df.na.fill(0)

# Select features and target variable
feature_columns = ['total_cases', 'new_cases', 'population']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(kenya_df).select('features', 'total_deaths')

### Step 2: Split the Data and Train the Model

# Split the data
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Train the model
lr = LinearRegression(featuresCol='features', labelCol='total_deaths')
lr_model = lr.fit(train_data)

# Make predictions
predictions = lr_model.transform(test_data)


### Step 3: Evaluate the Model

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="total_deaths", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {rmse}")


### Step 4: Visualize the Predictions

# Convert to Pandas DataFrame for visualization
pandas_df = predictions.select("total_deaths", "prediction").toPandas()

# Plot actual vs predicted values
plt.scatter(pandas_df['total_deaths'], pandas_df['prediction'])
plt.xlabel('Actual Total Deaths')
plt.ylabel('Predicted Total Deaths')
plt.title('Actual vs Predicted Total Deaths')
plt.show()
