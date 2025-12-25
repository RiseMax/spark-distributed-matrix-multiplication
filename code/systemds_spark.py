#spark-submit  --master spark://master:7077 --jars  /opt/miniconda3/envs/py310/lib/python3.10/site-packages/systemds/SystemDS.jar systemds_spark1.py
import numpy as np
import time
from pyspark.sql import SparkSession
from systemds.context import SystemDSContext

spark = SparkSession.builder \
    .appName("systemds-matmul-hdfs") \
    .master("spark://master:7077") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "6g") \
    .getOrCreate()


hdfs_path_A = "hdfs://master:9000/user/yourname/matrix_data/A_2000x1000.txt"
hdfs_path_B = "hdfs://master:9000/user/yourname/matrix_data/AT_1000x2000.txt"

A_rdd = spark.sparkContext.textFile(hdfs_path_A) \
    .map(lambda line: line.split(",")) \
    .map(lambda x: (int(x[0]), int(x[1]), float(x[2])))

B_rdd = spark.sparkContext.textFile(hdfs_path_B) \
    .map(lambda line: line.split(",")) \
    .map(lambda x: (int(x[0]), int(x[1]), float(x[2])))

M, K, N = 2000, 1000, 2000

A_np = np.zeros((M, K))
B_np = np.zeros((K, N))

for i, j, v in A_rdd.collect():
    A_np[i, j] = v

for i, j, v in B_rdd.collect():
    B_np[i, j] = v

with SystemDSContext() as sds:
    start = time.time()
    
    A_ds = sds.from_numpy(A_np)
    B_ds = sds.from_numpy(B_np)

    del A_np, B_np

    C_ds = A_ds @ B_ds
    result_sum = C_ds.sum().compute()
    
    systemds_time = time.time() - start
    print(f"SystemDS计算耗时: {systemds_time:.2f}秒")

spark.stop()