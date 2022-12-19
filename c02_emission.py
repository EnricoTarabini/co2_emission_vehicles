# Databricks notebook source
import os

co2 = spark.read.format('csv').option('header','true').option('inferSchema','true').load(f"file:{os.getcwd()}/co2.csv")
co2.display()

# COMMAND ----------

co2.count()

# COMMAND ----------

co2 = co2.na.drop()
co2.count()

# COMMAND ----------

co2.select('CO2 Emissions(g/km)').display()

# COMMAND ----------

display(co2.select('Vehicle Class','CO2 Emissions(g/km)'))

# COMMAND ----------

display(co2.select('Fuel Type','CO2 Emissions(g/km)'))

# COMMAND ----------

display(co2.select('Engine Size(L)','CO2 Emissions(g/km)'))

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder

# COMMAND ----------

stages = []

categoricalColumns = ['Make','Model','Vehicle Class','Transmission','Fuel Type']

# COMMAND ----------

for categoricalCol in categoricalColumns:

    indexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + 'Index')

    encoder = OneHotEncoder(inputCols=[indexer.getOutputCol()],outputCols=[categoricalCol + 'classVec'])

    stages += [indexer,  encoder]


# COMMAND ----------

numericCols = ['Engine Size(L)','Cylinders','Fuel Consumption City (L/100 km)','Fuel Consumption Hwy (L/100 km)']

assemblerInputs = [c + 'classVec' for c in categoricalColumns] + numericCols

assembler = VectorAssembler(inputCols = assemblerInputs, outputCol='features')

stages += [assembler]

# COMMAND ----------

stages

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(co2)

# COMMAND ----------

co2_transformed = pipelineModel.transform(co2)
co2_transformed.select('features','CO2 Emissions(g/km)').display()

# COMMAND ----------

co2_transformed = pipelineModel.transform(co2)
co2_transformed.select('features','CO2 Emissions(g/km)').display()

# COMMAND ----------

co2_train, co2_test = co2_transformed.randomSplit([0.7,0.3], seed = 0)

# COMMAND ----------

co2_train.count(), co2_test.count()

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(labelCol='CO2 Emissions(g/km)', subsamplingRate=0.8, numTrees=5)

# COMMAND ----------

rfModel = rf.fit(co2_train)

# COMMAND ----------

predictions = rfModel.transform(co2_test)
predictions.select('prediction','CO2 Emissions(g/km)','features').display()

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol='CO2 Emissions(g/km)', predictionCol='prediction')

r2 = evaluator.evaluate(predictions, {evaluator.metricName: 'r2'})

# COMMAND ----------

r2

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor

# with this regressor each tree is fit on the errors of the previous decision (instead of running in parallel)
gbt = GBTRegressor(labelCol='CO2 Emissions(g/km)', maxIter=50)

# COMMAND ----------

gbtModel = gbt.fit(co2_train)

# COMMAND ----------

predictions = gbtModel.transform(co2_test)
predictions.select('prediction','CO2 Emissions(g/km)','features').display()

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol='CO2 Emissions(g/km)', predictionCol='prediction')

r2 = evaluator.evaluate(predictions, {evaluator.metricName: 'r2'})

r2

# COMMAND ----------

import pyspark.sql.functions as F

predictions_with_residuals = predictions.withColumn('residual', (F.col('CO2 Emissions(g/km)') - F.col('prediction')))

# COMMAND ----------

display(predictions_with_residuals.agg({'residual':'mean'}))

# COMMAND ----------

display(predictions_with_residuals.select('Make','residual'))

# COMMAND ----------

display(predictions_with_residuals.select('residual'))

# COMMAND ----------


