from pyspark.sql import SparkSession
from pyspark import Broadcast
import numpy as np
import time
import random
from typing import Tuple, List
import os
import sys

# 定义MatrixEntry类（适配浮点型索引和值）
class MatrixEntry:
    def __init__(self, row: float, col: float, value: float):
        self.row = row
        self.col = col
        self.value = value
    
    def __repr__(self):
        return f"MatrixEntry({self.row:.2f}, {self.col:.2f}, {self.value:.6f})"

# ===================== 辅助函数：可视化与进度提示 =====================
def print_progress(msg: str, symbol: str = "="):
    """打印带格式的进度提示，增强可视化效果"""
    print(f"\n{symbol * 20} {msg} {symbol * 20}")

def preview_matrix_rdd(rdd, matrix_name: str, row_num: int = 5, col_num: int = 5):
    """预览矩阵RDD的前N行N列数据，可视化展示"""
    try:
        # 收集矩阵数据并转换为二维数组
        matrix_data = rdd.collect()
        if not matrix_data:
            print(f"{matrix_name} 矩阵无有效数据可预览")
            return
        
        # 提取行列索引和值，构建二维矩阵
        row_indices = sorted(list(set([int(x[0]) for x in matrix_data])))
        col_indices = sorted(list(set([int(x[1]) for x in matrix_data])))
        
        # 只预览前row_num行和col_num列
        preview_rows = row_indices[:row_num]
        preview_cols = col_indices[:col_num]
        
        # 构建预览矩阵
        preview_mat = np.zeros((len(preview_rows), len(preview_cols)))
        for (r, c, v) in matrix_data:
            r_int = int(r)
            c_int = int(c)
            if r_int in preview_rows and c_int in preview_cols:
                r_idx = preview_rows.index(r_int)
                c_idx = preview_cols.index(c_int)
                preview_mat[r_idx][c_idx] = v
        
        # 打印预览信息
        print(f"\n{matrix_name} 矩阵前 {len(preview_rows)} 行 {len(preview_cols)} 列 预览：")
        print("=" * 40)
        print(preview_mat.round(4))  # 保留4位小数，更清晰
        print("=" * 40)
        
    except Exception as e:
        print(f"{matrix_name} 矩阵预览失败: {e}")

def print_matrix_info(matrix_name: str, rows: int, cols: int, data_count: int):
    """打印矩阵详细信息"""
    print(f"\n【{matrix_name} 矩阵信息】")
    print(f"行数：{rows}")
    print(f"列数：{cols}")
    print(f"非零元素数：{data_count}")
    print(f"矩阵形状：{rows} × {cols}")

# ===================== 1. 环境配置（适配集群环境，兼容浮点型数据处理） =====================

# Spark Master地址（Docker Compose中master容器的地址）
SPARK_MASTER_URL = "spark://master:7077"
# Docker环境下Driver与Executor通信的地址
os.environ["SPARK_DRIVER_HOST"] = "host.docker.internal"

# 强制覆盖环境变量（Executor用容器Python，Driver用本地Python）
#os.environ["PYSPARK_PYTHON"] = DOCKER_PYTHON_EXE
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable 

# ===================== 2. 初始化Spark集群（自动使用最大资源，移除硬编码限制） =====================
def init_spark() -> SparkSession:
    # 集群资源配置：启用动态资源分配，同时启用外部Shuffle服务，解决依赖报错
    spark = SparkSession.builder \
        .appName("MatrixMultBroadcastTest-Cluster") \
        .master(SPARK_MASTER_URL) \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.driver.memory", "6g")\
        .config("spark.executor.memory", "6g")\
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    # 打印集群信息，验证资源配置
    print_progress("Spark集群初始化完成")
    print(f"Master URL: {spark.sparkContext.master}")
    print(f"Application ID: {spark.sparkContext.applicationId}")

    return spark

# ===================== 3. 读取HDFS矩阵文件（仅修改维度计算部分，其余不变） =====================
def read_matrix_from_file_csv(
    sc, 
    file_path: str,
    is_b_matrix: bool = False,  # 是否为B矩阵（用于broadcast优化）
    matrix_rows: int = 1,       # 手动指定矩阵行数（若为单行矩阵，默认1；可根据实际修改）
    matrix_cols: int = None     # 手动指定矩阵列数（默认None，自动从数据中获取）
) -> tuple:
    """
    从HDFS读取单行多列浮点型CSV文件，转换为二维矩阵RDD
    适配格式：CSV文件仅1行，包含多个浮点型数值（逗号分隔）
    返回：(矩阵RDD, 矩阵字典（仅B矩阵有）, 矩阵行数（整数）, 矩阵列数（整数）)
    """
    # 动态分区数：基于集群默认并行度，自动适配文件大小
    total_cores = sc.defaultParallelism
    partitions = total_cores  # 可根据文件大小调整为 total_cores * 2
    
    # 安全浮点转换辅助函数
    def safe_float_convert(val):
        try:
            return float(str(val).strip())
        except (ValueError, TypeError, Exception):
            return None
    
    # 读取并解析单行多列CSV文件
    try:
        print_progress(f"开始读取并解析文件: {os.path.basename(file_path)}", "-")
        # 第一步：读取原始文件，获取单行数据
        raw_rdd = sc.textFile(file_path, partitions)
        print(f"原始文件分区数：{raw_rdd.getNumPartitions()}")
        
        # 获取唯一行数据（过滤空行后仅保留一行）
        non_empty_rdd = raw_rdd.filter(lambda line: line.strip())
        line_count = non_empty_rdd.count()
        print(f"过滤空行后剩余行数：{line_count}")
        
        if line_count != 1:
            print(f"警告：CSV文件 {file_path} 不是单行数据，当前有 {line_count} 行")
            return sc.emptyRDD(), {}, 0, 0
        
        # 第二步：解析单行数据为浮点数组
        line_data = non_empty_rdd.first()  # 获取唯一行
        print(f"原始行数据长度（字符数）：{len(line_data)}")
        print(f"原始行数据前100个字符：{line_data[:100]}...")
        
        value_list = [safe_float_convert(val) for val in line_data.strip().split(",")]
        # 过滤转换失败的无效值
        valid_value_list = [v for v in value_list if v is not None]
        print(f"解析出浮点数值总数：{len(value_list)}")
        print(f"有效浮点数值数：{len(valid_value_list)}")
        
        # 第三步：确定矩阵行列数
        if matrix_cols is None:
            matrix_cols = len(valid_value_list)  # 自动获取列数（单行时，列数=数值个数）
        print(f"指定矩阵行数：{matrix_rows}，自动计算列数：{matrix_cols}")
        
        # 若指定了多行，需确保数值总数能被行数整除（按需调整）
        if len(valid_value_list) % matrix_rows != 0:
            print(f"警告：数值总数 {len(valid_value_list)} 无法被指定行数 {matrix_rows} 整除，将截断数据")
            valid_value_list = valid_value_list[:matrix_rows * matrix_cols]
            print(f"截断后数值总数：{len(valid_value_list)}")
        
        # 第四步：转换为（行索引, 列索引, 对应值）的RDD格式（兼容后续逻辑）
        matrix_data = []
        for row_idx in range(matrix_rows):
            for col_idx in range(matrix_cols):
                # 计算数值在一维列表中的索引
                val_idx = row_idx * matrix_cols + col_idx
                if val_idx < len(valid_value_list):
                    matrix_data.append((float(row_idx), float(col_idx), valid_value_list[val_idx]))
        
        # 转换为RDD
        matrix_rdd = sc.parallelize(matrix_data, partitions)
        data_count = matrix_rdd.count()
        
        # 验证有效数据
        if matrix_rdd.take(1):
            print(f"✅ 成功读取HDFS CSV文件: {file_path}")
            print(f"文件分区数：{matrix_rdd.getNumPartitions()}")
            print(f"有效矩阵数据：{matrix_rows} 行 × {matrix_cols} 列")
            print(f"矩阵元素总数：{data_count}")
            # 预览矩阵
            preview_matrix_rdd(matrix_rdd, os.path.basename(file_path).split(".")[0])
        else:
            print(f"❌ 警告：CSV文件 {file_path} 无有效数据")
            return sc.emptyRDD(), {}, 0, 0
            
    except Exception as e:
        print(f"❌ 读取/解析CSV文件失败: {e}")
        return sc.emptyRDD(), {}, 0, 0
    
    # 自动计算整数矩阵维度（直接使用指定/自动获取的行列数，无需再通过索引计算）
    try:
        final_rows = matrix_rows
        final_cols = matrix_cols
        print(f"✅ 自动识别矩阵维度: {final_rows} 行（整数） × {final_cols} 列（整数）")
        # 若需要获取索引范围（兼容原有日志）
        row_indices = [int(row_idx) for row_idx in range(final_rows)]
        col_indices = [int(col_idx) for col_idx in range(final_cols)]
        if row_indices and col_indices:
            print(f"原始浮点索引范围：行索引 [{min(row_indices)}, {max(row_indices)}]，列索引 [{min(col_indices)}, {max(col_indices)}]")
            
    except Exception as e:
        print(f"❌ 自动获取矩阵维度失败: {e}")
        final_rows, final_cols = 0, 0
    
    # B矩阵：按列存储为浮点型字典（原有逻辑不变）
    matrix_dict = {}
    if is_b_matrix and not matrix_rdd.isEmpty():
        try:
            print_progress(f"开始构建B矩阵列字典", "-")
            matrix_dict = matrix_rdd.map(lambda x: (int(round(x[1])), ((x[0], x[1]), x[2]))).groupByKey().collectAsMap()
            matrix_dict = {k: dict(v) for k, v in matrix_dict.items()}
            print(f"✅ 成功生成B矩阵Broadcast字典（包含 {len(matrix_dict)} 列数据）")
        except Exception as e:
            print(f"❌ 生成B矩阵Broadcast字典失败: {e}")
            matrix_dict = {}
    
    return matrix_rdd, matrix_dict, final_rows, final_cols


def read_matrix_from_file_txt(
    sc, 
    file_path: str,
    is_b_matrix: bool = False  # 是否为B矩阵（用于broadcast优化）
) -> tuple:
    """
    从HDFS读取浮点型矩阵文件，动态分区，自动计算整数矩阵维度
    适配格式：CSV文件，每行格式为(row_idx, col_idx, value)，均为浮点型
    返回：(矩阵RDD, 矩阵字典（仅B矩阵有）, 矩阵行数（整数）, 矩阵列数（整数）)
    优化点：提升读取速度，保留可视化进度展示
    """
    # 辅助函数：打印可视化进度
    def print_progress(msg: str, symbol: str = "-"):
        print(f"\n{symbol * 15} {msg} {symbol * 15}")
    
    # ===================== 优化1：提升分区策略，适配文件大小 =====================
    # 动态分区数：集群核心数的2~3倍（充分利用并行资源，避免小分区/大分区问题）
    total_cores = sc.defaultParallelism
    partitions = max(total_cores * 2, 8)  # 保底8个分区，避免核心数过少导致分区不足
    print_progress(f"初始化读取配置：分区数={partitions}", "=")

    # ===================== 优化2：设置HDFS读取优化参数，提升传输速度 =====================
    hadoop_conf = sc._jsc.hadoopConfiguration()
    # 增大读取缓冲区，提升大文件读取速度
    hadoop_conf.set("io.file.buffer.size", "131072")  # 128KB（默认4KB）
    # 设置超时时间，避免无效等待
    hadoop_conf.set("dfs.client.read.timeout", "300000")  # 5分钟超时
    hadoop_conf.set("dfs.socket.timeout", "300000")
    # 注释/删除短路读取配置（核心修复点）
    # hadoop_conf.set("dfs.client.read.shortcircuit", "true")  # 禁用该功能，避免配置缺失报错
    print("已配置HDFS读取优化参数：增大缓冲区、设置超时")

    # ===================== 优化3：轻量前置校验，避免无效全量读取 =====================
    print_progress("开始前置文件有效性校验")
    try:
        # 通过HDFS API快速判断文件是否存在/是否为文件（无需加载数据）
        path = sc._jvm.org.apache.hadoop.fs.Path(file_path)
        fs = path.getFileSystem(hadoop_conf)
        if not fs.exists(path):
            print(f"❌ 错误：HDFS文件不存在 -> {file_path}")
            return sc.emptyRDD(), {}, 0, 0
        if not fs.isFile(path):
            print(f"❌ 错误：指定路径不是文件 -> {file_path}")
            return sc.emptyRDD(), {}, 0, 0
        
        # 获取文件大小，可视化展示
        file_size = fs.getFileStatus(path).getLen()
        file_size_mb = round(file_size / 1024 / 1024, 2)
        print(f"✅ 文件校验通过：{file_path}（大小：{file_size_mb} MB）")
    except Exception as e:
        print(f"❌ 文件前置校验失败：{e}")
        return sc.emptyRDD(), {}, 0, 0

    # ===================== 优化4：优化RDD转换逻辑，减少冗余操作 =====================
    print_progress("开始读取并转换文件数据")
    try:
        # 优化：将filter和map操作合并，减少RDD依赖链；使用flatMap避免空数据
        def parse_line(line):
            line = line.strip()
            if not line:
                return []
            parts = line.split(",")
            if len(parts) != 3:
                return []
            try:
                # 一次性完成浮点转换，避免多次map
                return [(float(parts[0]), float(parts[1]), float(parts[2]))]
            except (ValueError, TypeError):
                return []
        
        # 读取文件：使用优化后的分区数，并行读取
        matrix_rdd = sc.textFile(file_path, partitions).flatMap(parse_line)
        
        # 优化：用take(1)替代默认take(1)，轻量验证数据是否有效（仅读取1条数据，不触发全量加载）
        sample_data = matrix_rdd.take(1)
        if not sample_data:
            print(f"❌ 警告：文件中无有效格式数据（需满足row,col,value格式）")
            return sc.emptyRDD(), {}, 0, 0
        
        print(f"✅ 成功读取文件并转换数据")
        print(f"📊 文件分区数（优化后）: {matrix_rdd.getNumPartitions()}")
        print(f"📌 数据样本：{sample_data[0]}")
    except Exception as e:
        print(f"❌ 读取HDFS文件或数据转换失败: {e}")
        return sc.emptyRDD(), {}, 0, 0

    # ===================== 优化5：并行计算矩阵维度，避免单线程collect阻塞 =====================
    print_progress("开始并行计算矩阵维度")
    try:
        # 优化：使用zipWithIndex避免全量collect，并行获取最大/最小索引
        # 步骤1：并行提取行/列索引（分布式计算，提升速度）
        row_rdd = matrix_rdd.map(lambda x: int(round(x[0]))).cache()  # 缓存避免重复计算
        col_rdd = matrix_rdd.map(lambda x: int(round(x[1]))).cache()
        
        # 步骤2：并行计算最大/最小索引（无需collect所有数据，大幅提升大文件速度）
        max_row = row_rdd.max() if not row_rdd.isEmpty() else -1
        min_row = row_rdd.min() if not row_rdd.isEmpty() else -1
        max_col = col_rdd.max() if not col_rdd.isEmpty() else -1
        min_col = col_rdd.min() if not col_rdd.isEmpty() else -1
        
        # 步骤3：计算矩阵维度，清理缓存
        matrix_rows = max_row + 1 if max_row >= 0 else 0
        matrix_cols = max_col + 1 if max_col >= 0 else 0
        row_rdd.unpersist()  # 释放缓存，避免内存占用
        col_rdd.unpersist()
        
        # 可视化维度信息
        if matrix_rows > 0 and matrix_cols > 0:
            print(f"✅ 自动识别矩阵维度: {matrix_rows} 行（整数） × {matrix_cols} 列（整数）")
            print(f"📊 原始浮点索引范围：")
            print(f"   行索引：[{min_row}, {max_row}]")
            print(f"   列索引：[{min_col}, {max_col}]")
        else:
            print(f"⚠️  警告：矩阵 {file_path} 无有效索引，返回0行0列")
            print(f"   行索引范围：[{min_row}, {max_row}]")
            print(f"   列索引范围：[{min_col}, {max_col}]")
            
    except Exception as e:
        print(f"❌ 自动获取矩阵维度失败: {e}")
        matrix_rows, matrix_cols = 0, 0

    # ===================== 优化6：优化B矩阵字典构建，减少冗余转换 =====================
    matrix_dict = {}
    if is_b_matrix and not matrix_rdd.isEmpty() and matrix_rows > 0 and matrix_cols > 0:
        print_progress("开始构建B矩阵Broadcast字典")
        try:
            # 优化：直接在map中完成整数转换，避免后续重复计算
            def map_to_col_key(x):
                col_idx = int(round(x[1]))
                key = (x[0], x[1])
                return (col_idx, (key, x[2]))
            
            # 按列分组，高效构建字典
            matrix_dict = matrix_rdd.map(map_to_col_key).groupByKey().collectAsMap()
            # 批量转换格式，提升效率
            matrix_dict = {k: dict(v) for k, v in matrix_dict.items()}
            
            print(f"✅ 成功生成B矩阵Broadcast字典")
            print(f"📊 字典信息：包含 {len(matrix_dict)} 列数据，覆盖矩阵所有列")
        except Exception as e:
            print(f"❌ 生成B矩阵Broadcast字典失败: {e}")
            matrix_dict = {}

    print_progress("文件读取与处理流程完成", "=")
    return matrix_rdd, matrix_dict, matrix_rows, matrix_cols
# ===================== 4. 原生矩阵乘法（全浮点型运算，RDD Join实现） =====================
def native_matrix_multiply(A_rdd, B_rdd, m: int, k: int, n: int) -> float:
    """
    原生矩阵乘法（基于RDD Join实现），全浮点型运算
    :param A_rdd: 矩阵A的RDD，格式为(row_idx, col_idx, value)（均为浮点型）
    :param B_rdd: 矩阵B的RDD，格式为(row_idx, col_idx, value)（均为浮点型）
    :param m: 矩阵A的行数（整数）
    :param k: 矩阵A的列数（整数，等于B的行数）
    :param n: 矩阵B的列数（整数）
    :return: 执行耗时（秒），失败返回0.0
    """
    if A_rdd.isEmpty() or B_rdd.isEmpty():
        print("❌ 原生矩阵乘法失败：A矩阵或B矩阵为空")
        return 0.0
    
    if k == 0:
        print("❌ 原生矩阵乘法失败：关联维度k为0（矩阵A的列数与矩阵B的行数不匹配）")
        return 0.0
    
    print_progress("开始执行原生矩阵乘法（RDD Join实现）")
    start_time = time.time()
    try:
        # 步骤1：A矩阵按列索引分组 (col_idx, (row_idx, float_value))
        A_by_col = A_rdd.map(lambda x: (x[1], (x[0], x[2])))
        # 步骤2：B矩阵按行索引分组 (row_idx, (col_idx, float_value))
        B_by_row = B_rdd.map(lambda x: (x[0], (x[1], x[2])))
        # 步骤3：Join后计算浮点乘积并聚合（A的列 = B的行）
        product = A_by_col.join(B_by_row) \
            .map(lambda x: ((x[1][0][0], x[1][1][0]), x[1][0][1] * x[1][1][1])) \
            .reduceByKey(lambda a, b: a + b)  # 浮点型累加
        # 触发计算（Spark自动分配资源执行）
        product_count = product.count()
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ 【原生矩阵乘法完成（全浮点型运算）】")
        print(f"结果矩阵元素数: {product_count}")
        print(f"执行耗时: {elapsed_time:.2f}s")
        print(f"矩阵维度：A({m}x{k}) × B({k}x{n}) = 结果({m}x{n})")
        
        # 预览结果矩阵
        preview_matrix_rdd(product, "原生乘法结果")
        
        return elapsed_time
    except Exception as e:
        print(f"❌ 原生矩阵乘法执行失败: {e}")
        return 0.0

# ===================== 5. Broadcast优化矩阵乘法（全浮点型运算，分区内高效计算） =====================
def broadcast_optimized_matrix_multiply(A_rdd, B_col_dict: dict, m: int, k: int, n: int) -> float:
    """
    Broadcast优化矩阵乘法，全浮点型运算，减少网络传输
    :param A_rdd: 矩阵A的RDD，格式为(row_idx, col_idx, value)（均为浮点型）
    :param B_col_dict: 矩阵B的列字典，格式为{col_idx: {(row_idx, col_idx): float_value}}
    :param m: 矩阵A的行数（整数）
    :param k: 矩阵A的列数（整数，等于B的行数）
    :param n: 矩阵B的列数（整数）
    :return: 执行耗时（秒），失败返回0.0
    """
    if not B_col_dict or A_rdd.isEmpty():
        print("❌ Broadcast优化矩阵乘法失败：B矩阵字典为空或A矩阵为空")
        return 0.0
    
    if k == 0:
        print("❌ Broadcast优化矩阵乘法失败：关联维度k为0（矩阵A的列数与矩阵B的行数不匹配）")
        return 0.0
    
    print_progress("开始执行Broadcast优化矩阵乘法（分区内高效计算）")
    start_time = time.time()
    try:
        # 广播B矩阵字典（仅传输一次到所有Executor）
        b_broadcast = A_rdd.context.broadcast(B_col_dict)
        print(f"✅ B矩阵字典已广播到Executor，字典大小：{len(B_col_dict)} 列")
        
        def compute_partition(iter):
            """分区内浮点型计算：按列查找B矩阵，减少无效遍历"""
            b_dict = b_broadcast.value
            result = {}
            # 遍历A矩阵的每个浮点型元素 (row_idx, col_idx, float_value)
            for (i, j, a_val) in iter:
                # 列索引转换为整数，匹配B矩阵字典的key
                j_int = int(round(j))
                if j_int not in b_dict:
                    continue
                # 遍历B矩阵中对应列的所有浮点型元素
                for (bk_row, bk_col), b_val in b_dict[j_int].items():
                    # 浮点型累加乘积：C[i][bk_col] += A[i][j] * B[bk_row][bk_col]
                    result_key = (i, bk_col)
                    result[result_key] = result.get(result_key, 0.0) + a_val * b_val
            return result.items()
        
        # 分区计算+聚合（全浮点型运算，Spark自动分配最大并行度）
        product = A_rdd.mapPartitions(compute_partition).reduceByKey(lambda a, b: a + b)
        product_count = product.count()
        # 释放Broadcast资源，避免内存泄漏
        b_broadcast.unpersist(blocking=True)
        print(f"✅ Broadcast资源已释放")
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ 【Broadcast优化矩阵乘法完成（全浮点型运算）】")
        print(f"结果矩阵元素数: {product_count}")
        print(f"执行耗时: {elapsed_time:.2f}s")
        print(f"矩阵维度：A({m}x{k}) × B({k}x{n}) = 结果({m}x{n})")
        
        # 预览结果矩阵
        preview_matrix_rdd(product, "Broadcast优化乘法结果")
        
        return elapsed_time
    except Exception as e:
        print(f"❌ Broadcast优化矩阵乘法执行失败: {e}")
        return 0.0

# ===================== 6. 测试主函数（支持选择乘法方式，全浮点型适配） =====================
def run_tests(choose_multiply: str = "both"):
    """
    测试主函数，支持选择乘法方式，全浮点型矩阵运算
    :param choose_multiply: 可选值："native"（仅原生乘法）、"broadcast"（仅优化乘法）、"both"（两者都执行）
    """
    # 初始化Spark集群
    spark = init_spark()
    sc = spark.sparkContext
    
    # 配置HDFS矩阵文件路径（替换为你的实际HDFS路径，已修正占位符）
    HDFS_MATRIX_DIR = "hdfs://master:9000/user/yourname/matrix_data"  # yourname替换为实际用户名sparkuser
    A_file = os.path.join(HDFS_MATRIX_DIR, "matrix_2000_5_A.txt")
    B_file = os.path.join(HDFS_MATRIX_DIR, "matrix_2000_5_B.txt")
    
    # 打印测试配置
    print_progress("开始矩阵乘法测试（全浮点型适配）")
    print(f"A矩阵HDFS路径: {A_file}")
    print(f"B矩阵HDFS路径: {B_file}")
    print(f"选择执行的乘法方式: {choose_multiply.upper()}")
    
    # 读取矩阵文件（全浮点型解析，自动计算整数维度）
    print_progress("开始读取矩阵A", "-")
    A_rdd, _, m, k = read_matrix_from_file_txt(
        sc, 
        A_file, 

    )  

    # 示例2：B.csv是单行50列矩阵（需作为B矩阵，开启is_b_matrix=True）
    print_progress("开始读取矩阵B", "-")
    B_rdd, B_col_dict, k_b, n = read_matrix_from_file_txt(
        sc, 
        B_file, 
        is_b_matrix=True, 

    )
    
    # 打印矩阵基本信息
    print_progress("矩阵读取完成，验证合法性")
    print_matrix_info("矩阵A", m, k, A_rdd.count() if not A_rdd.isEmpty() else 0)
    print_matrix_info("矩阵B", k_b, n, B_rdd.count() if not B_rdd.isEmpty() else 0)
    
    # 验证矩阵乘法合法性：A的列数必须等于B的行数（整数对比）
    if k != k_b:
        print(f"❌ 矩阵乘法不合法：A矩阵的列数({k}) ≠ B矩阵的行数({k_b})，无法进行乘法运算")
        spark.stop()
        return
    
    if A_rdd.isEmpty() or B_rdd.isEmpty():
        print(f"❌ 矩阵文件读取失败或为空，终止测试")
        spark.stop()
        return
    
    print(f"✅ 矩阵合法性验证通过")
    print(f"矩阵运算维度：A({m}x{k}) × B({k}x{n}) = 结果({m}x{n})")
    print_progress("开始执行矩阵乘法运算")
    
    native_time = 0.0
    broadcast_time = 0.0
    
    # 根据选择执行对应的乘法函数
    if choose_multiply in ["native", "both"]:
        native_time = native_matrix_multiply(A_rdd, B_rdd, m, k, n)
    
    if choose_multiply in ["broadcast", "both"]:
        broadcast_time = broadcast_optimized_matrix_multiply(A_rdd, B_col_dict, m, k, n)
    
    # 性能对比（仅当两者都执行且耗时有效时）
    print_progress("测试完成，汇总结果")
    if choose_multiply == "both" and broadcast_time > 0 and native_time > 0:
        speedup = (native_time - broadcast_time) / native_time * 100
        print(f"\n📊 【性能对比结果（全浮点型运算）】")
        print(f"原生矩阵乘法耗时: {native_time:.2f}s")
        print(f"Broadcast优化乘法耗时: {broadcast_time:.2f}s")
        print(f"性能提升比例: {speedup:.2f}%")
        print(f"结论：Broadcast优化比原生实现快 {native_time - broadcast_time:.2f} 秒")
    else:
        print(f"\n📊 【测试完成】")
        print(f"执行方式：{choose_multiply.upper()}")
        if native_time > 0:
            print(f"原生矩阵乘法耗时: {native_time:.2f}s")
        if broadcast_time > 0:
            print(f"Broadcast优化乘法耗时: {broadcast_time:.2f}s")
        print(f"无需进行性能对比（仅执行了单一乘法方式或耗时无效）")
    
    # 停止Spark会话，释放资源
    print_progress("停止Spark集群，释放资源")
    spark.stop()
    print(f"\n🎉 所有测试流程结束，Spark集群已正常停止")

# ===================== 入口函数（支持命令行参数或直接指定） =====================
if __name__ == "__main__":
    # Windows Docker特殊处理
    if os.name == "nt":
        os.environ["SPARK_DRIVER_HOST"] = "host.docker.internal"
    
    # 方式1：直接在代码中指定调用方式（可选："native"、"broadcast"、"both"）
    #run_tests(choose_multiply="native")  # 仅执行原生矩阵乘法
    run_tests(choose_multiply="broadcast")  # 仅执行Broadcast优化乘法
    #run_tests(choose_multiply="both")  # 两者都执行（默认）只改获取行列的部分，别的不用改 添加可视化运行