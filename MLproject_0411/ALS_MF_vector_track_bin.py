#This program is aim to create user factor and item factor based on only track rating data
#for further linear models like logistic regression and SVM

from pyspark import SparkContext,SparkConf
from pyspark.sql import HiveContext, Row
from pyspark.ml.recommendation import *

if __name__ == "__main__":
    conf = SparkConf().set('spark.executor.memory', '4g')
    sc = SparkContext(conf = conf)
    sqlContext = HiveContext(sc)

    # read file
    def read_hdfs_csv(sqlContext, filename, header='true'):
        csvreader = (sqlContext.read.format('com.databricks.spark.csv').options(header = header, inferschema='true'))
        return csvreader.load(filename)


    def write_hdfs_csv(df, filename):
        csvwriter = (df.write.format('com.databricks.spark.csv').options(header='true'))
        return csvwriter.save(filename)


    df_track = read_hdfs_csv(sqlContext, 'track_rate_bin.csv')


    def id_map(df, col):
        series = df.select(col).distinct().toPandas()[col]
        return series, dict(zip(series, range(len(series))))

    users, user_map = id_map(df_track, 'userID')
    items, item_map = id_map(df_track, 'itemID')

    def mapper(row):
        return Row(userID=user_map[row['userID']],
            itemID=item_map[row['itemID']],
            rating=row['rating_cat'])

    df_track_mapped = df_track.map(mapper).toDF(('userID', 'itemID', 'rating'))
    df_track_mapped = df_track_mapped.na.drop()

    als = ALS(userCol='userID',
            itemCol='itemID',
            ratingCol='rating')
    model_track = als.fit(df_track_mapped)

    model_track.userFactors.write.parquet('userfactor_track')
    model_track.itemFactors.write.parquet('itemfactor_track')

    sc.stop()

