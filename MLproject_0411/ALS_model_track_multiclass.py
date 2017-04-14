#try to build ALS model with pyspark.ml.recommendation
import sys
import itertools
import numpy as np
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext, Row
from pyspark import SparkConf, SparkContext
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator


if __name__ == "__main__":

    
    conf = SparkConf().set("spark.executor.memory", "8g").set("spark.yarn.executor.memoryOverhead", "2048")
    sc = SparkContext(conf = conf)
    sqlContext = HiveContext(sc)


    # read file
    def read_hdfs_csv(sqlContext, filename, header='true'):
        csvreader = (sqlContext.read.format('com.databricks.spark.csv').options(header = header, inferschema='true'))
        return csvreader.load(filename)


    def write_hdfs_csv(df, filename):
        csvwriter = (df.write.format('com.databricks.spark.csv').options(header='true'))
        return csvwriter.save(filename)

   
    df = read_hdfs_csv(sqlContext, 'track_rate_bin.csv')


    def id_map(df, col):
        series = df.select(col).distinct().toPandas()[col]
        return series, dict(zip(series, range(len(series))))

    users, user_map = id_map(df, 'userID')
    items, item_map = id_map(df, 'itemID')

    def mapper(row):
        return Row(userID=user_map[row['userID']],
            itemID=item_map[row['itemID']],
            rating=row[('rating_cat')])

    df_mapped = df.map(mapper).toDF()
    training, validation, test = df_mapped.randomSplit([0.6,0.2,0.2], seed = 12345)

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()
   
    als = ALS(userCol="userId", itemCol="itemId", ratingCol="rating")
    model = als.fit(training)
    valPredictions = model.transform(validation)
    val_predictions = valPredictions\
                  .withColumn("rating", valPredictions.rating.cast("double"))\
                  .withColumn("prediction", valPredictions.prediction.cast("double"))
    #evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    validation_score = sqrt(val_predictions.map(lambda x: (x[1] - x[3]) ** 2).reduce(add) / float(numValidation))

    testPredictions = model.transform(test)
    test_predictions = testPredictions\
                       .withColumn("rating", testPredictions.rating.cast("double"))\
                       .withColumn("prediction", testPredictions.prediction.cast("double"))
    test_score = sqrt(test_predictions.map(lambda x: (x[1] - x[3]) ** 2).reduce(add) / float(numTest))
    
    #write_hdfs_csv(val_predictions, "validation_prediction.csv")
    #write_hdfs_csv(test_predictions, "test_prediction.csv")  
    print "Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest)
    print "Accuracy (validation) = %f for the model trained with " % validation_score
    print "Accuracy (validation) = %f for the model trained with " % test_score


    # clean up
    sc.stop()
