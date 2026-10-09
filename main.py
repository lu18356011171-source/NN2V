import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from data_utils import load_and_preprocess_data
# 导入升级后的多尺度模型
from model import SpectralBSN


def physics_informed_loss(y_pred, y_true, lambda_reg):
    """
    物理融合损失函数 = MSE (逼近数据) + Lambda * 二阶导数L1惩罚 (物理平滑)
    """
    # 1. 基础重建损失
    mse_loss = nn.MSELoss()(y_pred, y_true)

    # 2. 物理平滑损失 (二阶导数的离散近似: y[i+1] - 2y[i] + y[i-1])
    # 由于两端会缺少点，我们只计算中间部分的二阶导数
    d2y = y_pred[:, :, 2:] - 2 * y_pred[:, :, 1:-1] + y_pred[:, :, :-2]
    # 取绝对值(abs)的平均值(torch.mean)，强迫网络压制高频小毛刺
    smoothness_loss = torch.mean(torch.abs(d2y))

    # 总损失
    total_loss = mse_loss + lambda_reg*smoothness_loss
    return total_loss, mse_loss, smoothness_loss


def main():
    # ==========================================
    # 1. 数据加载 (请替换为你的实际路径)
    # ==========================================
    data_filepath = r"C:\Users\LYM\Desktop\长光所研二资料\小论文资料\气体\氢气\氢气1000除1000.xlsx"

    print(f"正在加载数据...")
    x_input, v_min, v_max, wavelengths = load_and_preprocess_data(data_filepath)
    if x_input is None: return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_input = x_input.to(device)

    # ==========================================
    # 2. 模型与优化器
    # ==========================================
    model = SpectralBSN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==========================================
    # 3. 物理自监督训练循环
    # ==========================================
    epochs = 400
    # lambda_reg 控制平滑度。如果曲线依然毛躁，调大它(如0.1)；如果拉曼峰被压低了，调小它(如0.01)
    lambda_reg = 0.075

    print(f"开始训练 MS-PI-BSN (引入物理约束 \u03BB={lambda_reg})...")
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_input)

        # 使用定制的物理损失函数
        loss, mse, smooth = physics_informed_loss(y_pred, x_input, lambda_reg=lambda_reg)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] | Total Loss: {loss.item():.6f} | MSE: {mse.item():.6f} | Smooth: {smooth.item():.6f}")

    # ==========================================
    # 4. 推理与可视化
    # ==========================================
    model.eval()
    with torch.no_grad():
        denoised_tensor = model(x_input)

    denoised_intensities = denoised_tensor.cpu().numpy().squeeze() * (v_max - v_min) + v_min
    original_intensities = x_input.cpu().numpy().squeeze() * (v_max - v_min) + v_min

    # ==========================================
    # 5. 导出处理后的数据 (新增：导出为 TXT 和 Excel)
    # ==========================================
    print("正在导出去噪后的光谱数据...")

    # 自动获取输入文件所在的文件夹路径和原始文件名
    output_dir = os.path.dirname(data_filepath)
    base_name = os.path.splitext(os.path.basename(data_filepath))[0]

    # --- 方式 A：导出为 TXT (逗号分隔，极其适合 Origin 直接拖拽导入) ---
    txt_output_path = os.path.join(output_dir, f"{base_name}_pred+ture.txt")
    # 将一维的波长和强度数组按列拼成二维矩阵
    export_matrix = np.column_stack((wavelengths, denoised_intensities))
    # fmt='%.6f' 保留6位小数，header 设置表头
    np.savetxt(txt_output_path, export_matrix, delimiter=',', fmt='%.6f',
               header='Wavelength,Intensity', comments='')
    print(f" [成功] TXT 数据已保存至: {txt_output_path}")

if __name__ == "__main__":
    main()
