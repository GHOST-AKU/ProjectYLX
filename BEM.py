import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time

# --- 配置 ---
CSV_FILE_PATH = '85.csv'  # <--- 请将这里替换为你的CSV文件路径
TARGET_COLUMN = '平均数'         # 目标评分列的名称
TEST_SIZE = 0.2                 # 测试集比例
RANDOM_STATE = 42               # 随机种子，保证结果可复现

try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    matplotlib.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
except Exception as e:
    print(f"设置中文字体失败: {e}. 特征名称可能显示不正常。")
    print("请确保系统中安装了 'SimHei' 字体，或尝试其他中文字体。")

# --- 1. 读取数据 ---
try:
    data = pd.read_csv(CSV_FILE_PATH, encoding='gbk')
    print(f"成功读取文件: {CSV_FILE_PATH}")
    print("数据前5行:")
    print(data.head())
    print("\n数据信息:")
    data.info()
except FileNotFoundError:
    print(f"错误: 文件未找到 '{CSV_FILE_PATH}'")
    exit()
except Exception as e:
    print(f"读取CSV文件时出错: {e}")
    exit()

# --- 2. 准备特征和目标 ---
if TARGET_COLUMN not in data.columns:
    print(f"错误: 目标列 '{TARGET_COLUMN}' 在CSV文件中未找到。")
    print(f"可用的列: {data.columns.tolist()}")
    exit()

# 分离特征 (X) 和目标 (y)
y = data[TARGET_COLUMN]
X = data.drop(columns=[TARGET_COLUMN])

# 选择数值类型的特征列进行训练
X_numeric = X.select_dtypes(include=np.number)

# 检查是否有数值特征
if X_numeric.empty:
    print("错误：找不到任何数值类型的特征列用于训练。")
    exit()

print(f"\n使用的特征列 ({len(X_numeric.columns)}):")
print(X_numeric.columns.tolist())
print(f"\n目标列: {TARGET_COLUMN}")

# 处理缺失值（简单填充为均值）
# 你也可以选择其他策略，如填充中位数或删除包含NaN的行
if X_numeric.isnull().sum().sum() > 0:
    print("\n检测到缺失值，将使用均值填充...")
    X_numeric = X_numeric.fillna(X_numeric.mean())
    print("缺失值处理完毕。")

# --- 3. 划分训练集和测试集 ---
X_train, X_test, y_train, y_test = train_test_split(
    X_numeric, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\n训练集大小: {X_train.shape[0]}样本")
print(f"测试集大小: {X_test.shape[0]}样本")

# --- 4. 训练随机森林模型 ---
print("\n开始训练随机森林回归模型...")
start_time_rf = time.time()
rf_reg = RandomForestRegressor(
    n_estimators=100,             # 树的数量
    max_depth=2,                  # 树的最大深度
    random_state=RANDOM_STATE,
    n_jobs=-1                     # 使用所有可用CPU核心
)

rf_reg.fit(X_train, y_train)
rf_train_time = time.time() - start_time_rf
print(f"随机森林模型训练完成。训练时间: {rf_train_time:.2f}秒")

# --- 5. 训练XGBoost模型 ---
print("\n开始训练XGBoost回归模型...")
start_time_xgb = time.time()
xgb_reg = xgb.XGBRegressor(
    objective='reg:squarederror', # 回归任务，使用平方误差
    n_estimators=100,             # 树的数量
    learning_rate=0.1,            # 学习率
    max_depth=2,                  # 树的最大深度
    subsample=0.8,                # 训练每棵树时样本的采样比例
    colsample_bytree=0.8,         # 训练每棵树时特征的采样比例
    random_state=RANDOM_STATE,
    n_jobs=-1                     # 使用所有可用CPU核心
)

xgb_reg.fit(X_train, y_train)
xgb_train_time = time.time() - start_time_xgb
print(f"XGBoost模型训练完成。训练时间: {xgb_train_time:.2f}秒")

# --- 6. 评估模型并比较 ---
# 随机森林预测
rf_pred = rf_reg.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)

# XGBoost预测
xgb_pred = xgb_reg.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(xgb_mse)
xgb_r2 = r2_score(y_test, xgb_pred)

# 简化评估输出部分
print("\n模型评估对比:")
print("-" * 70)
print(f"指标         随机森林        XGBoost        提升百分比")
print("-" * 70)
print(f"MSE          {rf_mse:.4f}        {xgb_mse:.4f}        {(rf_mse - xgb_mse) / rf_mse * 100:.2f}%")
print(f"RMSE         {rf_rmse:.4f}        {xgb_rmse:.4f}        {(rf_rmse - xgb_rmse) / rf_rmse * 100:.2f}%")
print(f"R²           {rf_r2:.4f}        {xgb_r2:.4f}        {(xgb_r2 - rf_r2) / abs(rf_r2) * 100:.2f}%")
print(f"训练时间(秒)  {rf_train_time:.2f}        {xgb_train_time:.2f}        {(rf_train_time - xgb_train_time) / rf_train_time * 100:.2f}%")
print("-" * 70)

# # --- 7. 获取并对比特征重要性 ---
# print("\n计算特征重要性...")
# rf_importances = rf_reg.feature_importances_
# xgb_importances = xgb_reg.feature_importances_
# feature_names = X_numeric.columns

# # 创建一个包含特征名称和两个模型重要性的DataFrame，并排序
# feature_importance_df = pd.DataFrame({
#     'feature': feature_names,
#     'rf_importance': rf_importances,
#     'xgb_importance': xgb_importances
# })

# # 按XGBoost重要性排序
# feature_importance_df = feature_importance_df.sort_values(by='xgb_importance', ascending=False)

# print("\n特征重要性对比:")
# print(feature_importance_df)

# # --- 8. 绘制特征重要性对比图 ---
# plt.figure(figsize=(14, max(8, len(feature_names) // 2))) # 动态调整图形大小

# # 创建条形图
# x = np.arange(len(feature_names))
# width = 0.35  # 条形宽度

# plt.barh(x - width/2, feature_importance_df['rf_importance'], width, label='随机森林', color='lightgreen')
# plt.barh(x + width/2, feature_importance_df['xgb_importance'], width, label='XGBoost', color='skyblue')

# plt.yticks(x, feature_importance_df['feature'])
# plt.xlabel('特征重要性')
# plt.title('随机森林 vs XGBoost 特征重要性对比')
# plt.legend()
# plt.gca().invert_yaxis()  # 让最重要的特征显示在顶部
# plt.tight_layout()  # 调整布局防止标签重叠

# # 保存特征对比图
# plot_filename_compare = 'feature_importance_comparison.png'
# try:
#     plt.savefig(plot_filename_compare)
#     print(f"\n特征重要性对比图已保存为: {plot_filename_compare}")
# except Exception as e:
#     print(f"保存对比图时出错: {e}")

# # 单独绘制XGBoost特征重要性图
# plt.figure(figsize=(12, max(6, len(feature_names) // 2)))
# plt.barh(feature_importance_df['feature'], feature_importance_df['xgb_importance'], color='skyblue')
# plt.xlabel('特征重要性')
# plt.ylabel('特征名称')
# plt.title('XGBoost模型特征重要性')
# plt.gca().invert_yaxis()
# plt.tight_layout()

# plot_filename_xgb = 'feature_importance_xgboost.png'
# try:
#     plt.savefig(plot_filename_xgb)
#     print(f"XGBoost特征重要性图已保存为: {plot_filename_xgb}")
# except Exception as e:
#     print(f"保存XGBoost图时出错: {e}")

# plt.show()

# print("\n脚本执行完毕。")