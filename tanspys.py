# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 12:06:44 2019

@author: 687338
"""
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('trnsformerfpy').getOrCreate()
trns_df=spark.read.csv('C:\\Users\\687338\\Desktop\\Datasets\\trns1cs.csv',header='TRUE', inferSchema='TRUE')
trns_df.printSchema()
trns_df.head(10)
trns_df.columns
#from pyspark.sql.functions import col
#trns_df=trns_df.withColumn('Total Gas',col('Total Gas').cast('Int'))
#trns_df.printSchema()

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
assembler=VectorAssembler(inputCols=[
 'Time DGOA to problem',
 'kVA (1)',
 'kVA (2)',
 'Age at DGOA test',
 'H2',
 'CH4',
 'CO',
 'CO2',
 'C2H6',
 'C2H4',
 'C2H2',
 'Total Gas'],outputCol='features')
output_data=assembler.setHandleInvalid('skip').transform(trns_df)
output_data.printSchema()


train_data,test_data=output_data.randomSplit([.8,.2],seed=1234)

from pyspark.ml.classification import LogisticRegression
lr=LogisticRegression(labelCol='Failure',featuresCol='features',maxIter=10,regParam=.3)


train_model=lr.fit(train_data)

predictions = train_model.transform(test_data)
predictions.printSchema()

predictions.select('Failure','prediction','probability').show(5)

cm=predictions.select('Failure','prediction')
print(cm.filter(cm.Failure == cm.prediction).count() / cm.count())

