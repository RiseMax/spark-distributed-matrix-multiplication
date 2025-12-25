# spgemm_final.py
# ----------------------------------------
# Sparse Matrix Multiplication (SpGEMM)
# Optimization: map-side aggregation
# Timing: use Spark UI (Task Time), NOT Python
# ----------------------------------------

import sys
import time                      # ←【新增】
from pyspark.sql import SparkSession

# ==================================================
# 1. 在这里直接指定数据集路径（你只需要改这里）
# ==================================================
A_PATH = "hdfs:///user/yourname/matrix_data/matrix_1000_5_A.txt"     # A: i,k,value
B_PATH = "hdfs:///user/yourname/matrix_data/matrix_1000_5_B.txt"     # B: k,j,value
OUT_PATH = "None"             # 或 "/path/to/output/C"

# 并行度设置（可根据集群规模调整）
PARTITIONS = 64

# ==================================================
# 工具函数
# ==================================================
def parse_triplet(line):
    # 输入格式: row,col,value
    parts = line.strip().split(",")
    if len(parts) != 3:
        return None
    return int(parts[0]), int(parts[1]), float(parts[2])

def local_aggregate(iterator):
    """
    map-side aggregation:
    在每个 partition 内，对 (i,j) 先做局部累加
    """
    acc = {}
    for _, ((i, a), (j, b)) in iterator:
        key = (i, j)
        acc[key] = acc.get(key, 0.0) + a * b
    for k, v in acc.items():
        yield k, v

# ==================================================
# 主程序
# ==================================================
def main():
    spark = SparkSession.builder \
        .appName("SpGEMM-MapSideAggregation") \
        .getOrCreate()

    sc = spark.sparkContext

    # ------------------------------------------------
    # 2. 读入数据（IO 阶段，不计入计算时间）
    # ------------------------------------------------
    # A: (i,k,val) → (k,(i,val))
    A0 = (sc.textFile(A_PATH)
            .map(parse_triplet)
            .filter(lambda x: x is not None)
            .map(lambda t: (t[1], (t[0], t[2]))))

    # B: (k,j,val) → (k,(j,val))
    B0 = (sc.textFile(B_PATH)
            .map(parse_triplet)
            .filter(lambda x: x is not None)
            .map(lambda t: (t[0], (t[1], t[2]))))

    # ------------------------------------------------
    # 3. cache + materialize（剥离 IO 影响）
    # ------------------------------------------------
    A = A0.partitionBy(PARTITIONS).persist()
    B = B0.partitionBy(PARTITIONS).persist()
    A.count()
    B.count()

    # ------------------------------------------------
    # 4. 矩阵计算（计时只覆盖这一段）
    # ------------------------------------------------
    sc.setJobGroup(
        "MATMUL_COMPUTE",
        "Sparse matrix multiplication with map-side aggregation",
        interruptOnCancel=True
    )

    C = (A.join(B)                       # (k, ((i,a),(j,b)))
           .mapPartitions(local_aggregate)
           .reduceByKey(lambda x, y: x + y))

    t0 = time.perf_counter()             # ←【新增：开始计时】
    nnz_c = C.count()                    # 触发真正计算
    t1 = time.perf_counter()             # ←【新增：结束计时】

    sc.clearJobGroup()

    print(f"[RESULT] nnz(C) = {nnz_c}")
    print(f"[TIME] compute_time = {t1 - t0:.6f} seconds")   # ←【新增】

    # ------------------------------------------------
    # 5. 输出结果（IO，不计入计算时间）
    # ------------------------------------------------
    if OUT_PATH.lower() != "none":
        (C.map(lambda kv: f"{kv[0][0]},{kv[0][1]},{kv[1]}")
          .saveAsTextFile(OUT_PATH))

    spark.stop()

# ==================================================
if __name__ == "__main__":
    main()
