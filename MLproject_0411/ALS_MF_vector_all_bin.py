#This program is aim to create user factor and item factor based on only track rating data
#for further linear models like logistic regression and SVM

from pyspark import SparkContext,SparkConf
from pyspark.sql import HiveContext, Row
from pyspark.ml.recommendation import *

if __name__ == "__main__":
    conf = SparkConf().set("spark.executor.memory", "8g").set("spark.yarn.executor.memoryOverhead", "2048")
    sc = SparkContext(conf=conf)
    sqlContext = HiveContext(sc)

    # read file
    def read_hdfs_csv(sqlContext, filename, header='true'):
        csvreader = (sqlContext.read.format('com.databricks.spark.csv').options(header = header, inferschema='true'))
        return csvreader.load(filename)


    def write_hdfs_csv(df, filename):
        csvwriter = (df.write.format('com.databricks.spark.csv').options(header='true'))
        return csvwriter.save(filename)


    df_all = read_hdfs_csv(sqlContext, 'train_merge_bin.csv')


    def id_map(df, col):
        series = df.select(col).distinct().toPandas()[col]
        return series, dict(zip(series, range(len(series))))

    users, user_map = id_map(df_all, 'userID')
    items, item_map = id_map(df_all, 'itemID')

    def mapper(row):
        return Row(userID=user_map[row['userID']],
            itemID=item_map[row['itemID']],
            rating=row['rating_cat'])

    df_all_mapped = df_all.map(mapper).toDF()
    df_all_mapped = df_all_mapped.na.drop()
   
    als = ALS(
            rank=50,
            userCol='userID',
            itemCol='itemID',
            ratingCol='rating')
    model_all = als.fit(df_all_mapped)
    
    model_all.userFactors.write.parquet('userfactor_all')
    model_all.itemFactors.write.parquet('itemfactor_all')

    sc.stop()


