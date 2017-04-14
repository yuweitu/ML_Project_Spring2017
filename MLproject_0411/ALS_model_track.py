#build ALS model with pyspark.ml.recommendation
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

if __name__ == "__main__":

    
    conf = SparkConf().set("spark.executor.memory", "8g").set("spark.yarn.executor.memoryOverhead", "2048")
    sc = SparkContext(conf = conf)
    sc.setCheckpointDir('checkpoint/')
    sqlContext = HiveContext(sc)


    f = open('model_evaluation.txt', 'w')


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
            rating=row['rating_cat'])

    df_mapped = df.map(mapper).toDF(("userID","itemID", "rating"))
    training, validation, test = df_mapped.randomSplit([0.6,0.2,0.2], seed = 12345)

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    print "Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest)


    ranks = [50, 100]
    alphas = [1]
    maxIters = [10]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestalpha = -1.0
    bestmaxIter = -1

    for rank, alpha, maxIter in itertools.product(ranks,alphas, maxIters):
        als = ALS(rank = rank, maxIter = maxIter, userCol="userId", itemCol="itemId", ratingCol="rating")
        als.checkpointInterval = 2
        model = als.fit(training)
        rawPredictions = model.transform(validation)
        predictions = rawPredictions\
            .withColumn("rating", rawPredictions.rating.cast("double"))\
            .withColumn("prediction", rawPredictions.prediction.cast("double"))
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        validationRmse = evaluator.evaluate(predictions)
        print "RMSE (validation) = %f for the model trained with " % validationRmse + \
              "rank = %d, alpha = %.1f, and maxIter = %d." % (rank, alpha, maxIter)
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestalpha = alpha
            bestmaxIter = maxIter

    rawPredictions = model.transform(test)
    test_predictions = rawPredictions\
            .withColumn("rating", rawPredictions.rating.cast("double"))\
            .withColumn("prediction", rawPredictions.prediction.cast("double"))
    testRmse = evaluator.evaluate(test_predictions)

    # evaluate the best model on the test set
    print "The best model was trained with rank = %d and alpha = %.1f, " % (bestRank, bestalpha) \
      + "and maxIter = %d, and its RMSE on the test set is %f." % (bestmaxIter, testRmse)

    f.close()

    # clean up
    sc.stop()

