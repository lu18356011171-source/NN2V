import torch
import numpy as np
import pandas as pd
import os


def load_and_preprocess_data(filepath):
    """
    加载光谱数据，支持 .txt, .csv, .xlsx, .xls
    进行 Min-Max 归一化，并转换为 PyTorch Tensor
    """
    try:
        # 获取文件后缀名并转换为小写
        ext = os.path.splitext(filepath)[1].lower()

        if ext in ['.txt', '.csv']:
            # 读取 txt/csv，假设是逗号分隔
            data = np.loadtxt(filepath, delimiter=',')

        elif ext in ['.xlsx', '.xls']:
            # 使用 pandas 读取 Excel
            df = pd.read_excel(filepath, header=None)
            # 鲁棒性处理：强制转换为数值类型。如果 Excel 首行是文字表头，会被转为 NaN，然后 dropna 剔除
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            data = df.values

        else:
            print(f"不支持的文件格式: {ext}，仅支持 txt, csv, xlsx, xls")
            return None, None, None, None

        # 提取波长（第一列）和强度（第二列）
        wavelengths = data[:, 0]
        intensities = data[:, 1]

        # 极值归一化 (Min-Max Scaling 到 0-1 之间)
        v_min = np.min(intensities)
        v_max = np.max(intensities)

        # 防止分母为 0 导致报错
        if v_max == v_min:
            intensities_norm = intensities - v_min
        else:
            intensities_norm = (intensities - v_min) / (v_max - v_min)

        # 转换为 PyTorch 张量
        # 增加维度：[长度] -> [1 (Batch), 1 (Channel), 长度 (Sequence_Length)]
        tensor_data = torch.tensor(intensities_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        return tensor_data, v_min, v_max, wavelengths

    except Exception as e:
        print(f"读取数据失败，请检查文件路径和内容格式:\n{e}")
        return None, None, None, None