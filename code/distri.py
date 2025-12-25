from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
import time
import os
import argparse

DOCKER_PYTHON_EXE = "/opt/miniconda3/envs/py310/bin/python"

spark = SparkSession.builder \
    .appName("MatrixTestFromHDFS") \
    .config("spark.executorEnv.PYSPARK_PYTHON", DOCKER_PYTHON_EXE) \
    .config("spark.pyspark.driver.python", DOCKER_PYTHON_EXE) \
    .config("spark.pyspark.python", DOCKER_PYTHON_EXE) \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

def main(a_file_path, b_file_path):
    sc = spark.sparkContext
    # --- 数据读取部分保持不变 ---
    # 从 HDFS 读矩阵 A
    # rddA = sc.textFile("hdfs:///user/yourname/matrix/A_1000_10.txt") \
    #     .map(lambda line: line.split(",")) \
    #     .map(lambda parts: MatrixEntry(int(parts[0]), int(parts[1]), float(parts[2])))
    rddA = sc.textFile("hdfs:///user/yourname/matrix_data/matrix_1000_5_A.txt") \
            .map(lambda line: line.split(",")) \
            .map(lambda parts: MatrixEntry(int(parts[0]), int(parts[1]), float(parts[2])))
    # 同样读矩阵 B
    # rddB = sc.textFile("hdfs:///user/yourname/matrix/B_1000_10.txt") \
    #     .map(lambda line: line.split(",")) \
    #     .map(lambda parts: MatrixEntry(int(parts[0]), int(parts[1]), float(parts[2])))
    rddB = sc.textFile("hdfs:///user/yourname/matrix_data/matrix_1000_5_B.txt") \
            .map(lambda line: line.split(",")) \
            .map(lambda parts: MatrixEntry(int(parts[0]), int(parts[1]), float(parts[2])))
    # 转成 CoordinateMatrix
    coordA = CoordinateMatrix(rddA)
    coordB = CoordinateMatrix(rddB)
    start_time = time.time()

    # 转成 BlockMatrix 方便乘法
    blockA = coordA.toBlockMatrix().cache()
    blockB = coordB.toBlockMatrix().cache()

    # 执行分布式乘法 (此时尚未真正执行，只是建立了 DAG)
    product = blockA.multiply(blockB)

    # --- 修改部分 ---

    # 输出 shape (这通常会触发轻量级的元数据计算，但不会触发完整矩阵乘法)
    print("Result matrix dimensions: rows =", product.numRows(), "cols =", product.numCols())

    # 1. 将 BlockMatrix 转回 CoordinateMatrix 以访问底层的 RDD
    #    这一步是分布式的，数据依然在 Executor 上
    result_coords = product.toCoordinateMatrix()

    # 2. 使用 Action 算子触发计算
    #    .entries 访问底层的 RDD[MatrixEntry]
    #    .count() 会强制整个矩阵乘法流程执行，并统计结果中非零元素的个数
    #    结果仅仅是一个整数返回给 Driver，不会撑爆内存
    nnz_count = result_coords.entries.count()
    print(f"Computation triggered. Total non-zero entries: {nnz_count}")

    # 3. (可选) 如果想确认结果是否正确，只拉取前 5 条数据看一眼
    #    take(n) 也是 Action，会返回少量数据
    sample_data = result_coords.entries.take(5)
    print("Sample first 5 entries:", sample_data)

    # 注意：千万不要再调用 product.toLocalMatrix() 除非你确定结果矩阵极小
    end_time = time.time()
    duration = end_time - start_time
    print(f"Computation Finished!")
    print(f"Total Non-Zero Entries: {nnz_count}")
    print(f"Time Taken: {duration:.4f} seconds")
    spark.stop()

if __name__ == "__main__":
    HDFS_MATRIX_DIR = "hdfs:///user/yourname/matrix_data"
    parser = argparse.ArgumentParser(description="从文件读取稀疏矩阵并做乘法，格式 row,col,value")
    parser.add_argument("--a-path", default="matrix_1000_5_A.txt", help="矩阵 A 的路径")
    parser.add_argument("--b-path", default="matrix_1000_5_B.txt", help="矩阵 B 的路径")
    args = parser.parse_args()
    a_file = os.path.join(HDFS_MATRIX_DIR, "matrix_5000_5_A.txt")
    b_file = os.path.join(HDFS_MATRIX_DIR, "matrix_5000_5_B.txt")
    main(a_file, b_file)
