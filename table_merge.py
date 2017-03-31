import sys
from pyspark import SparkContext
from csv import reader
from pyspark.sql import HiveContext
from pyspark.sql import SQLContext

if __name__ == "__main__":

    sc = SparkContext()
    sqlContext = SQLContext(sc)

    # read file
    def read_hdfs_csv(sqlContext, filename, header='false'):
        csvreader = (sqlContext.read.format('com.databricks.spark.csv').options(header = header, inferschema='true'))
        return csvreader.load(filename)


    def write_hdfs_csv(df, filename):
        csvwriter = (df.write.format('com.databricks.spark.csv').options(header='true'))
        return csvwriter.save(filename)

    ID_table = read_hdfs_csv(sqlContext, 'ID_table.csv').toDF('itemID','label')
    train_data = read_hdfs_csv(sqlContext, 'train_data.csv').toDF('userID','itemID','rating')

    sqlContext.registerDataFrameAsTable(ID_table, 'ID')
    sqlContext.registerDataFrameAsTable(train_data, 'train')

    sql_1 = '''
        SELECT train.*, ID.label
        FROM train
        INNER JOIN ID
        ON train.itemID = ID.itemID
        '''
    sql_2 = '''
        SELECT train.*, ID.label
        FROM train
        INNER JOIN ID
        ON train.itemID = ID.itemID
        WHERE ID.label = 'track'
        '''
    sql_3 = '''
        SELECT train.*, ID.label
        FROM train
        INNER JOIN ID
        ON train.itemID = ID.itemID
        WHERE ID.label = 'artist'
        '''
    sql_4 = '''
        SELECT train.*, ID.label
        FROM train
        INNER JOIN ID
        ON train.itemID = ID.itemID
        WHERE ID.label = 'album'
        '''
    sql_5 = '''
        SELECT train.*, ID.label
        FROM train
        INNER JOIN ID
        ON train.itemID = ID.itemID
        WHERE ID.label = 'genre'
        '''
    train_merge = sqlContext.sql(sql_1)
    track_rate = sqlContext.sql(sql_2)
    artist_rate = sqlContext.sql(sql_3)
    album_rate = sqlContext.sql(sql_4)
    genre_rate = sqlContext.sql(sql_5)


    write_hdfs_csv(train_merge, 'train_merge.csv')
    write_hdfs_csv(track_rate, 'track_rate.csv')
    write_hdfs_csv(artist_rate, 'artist_rate.csv')
    write_hdfs_csv(album_rate, 'album_rate.csv')
    write_hdfs_csv(genre_rate, 'genre_rate.csv')
    exit()
