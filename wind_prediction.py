"""
风速预测项目 - 完整代码
商学院机器学习课程期末大作业
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 第一部分：加载和合并数据
# ============================================
print("="*50)
print("第一步：加载数据")
print("="*50)

# 定义文件路径
data_dir = "./data/"
heights = ['10m', '50m', '100m']

# 用于存储三个高度的数据框
dfs = {}

for height in heights:
    print(f"正在加载 {height} 数据...")
    # 读取三个文件
    train_df = pd.read_parquet(f"{data_dir}{height}/train-00000-of-00001.parquet")
    val_df = pd.read_parquet(f"{data_dir}{height}/val-00000-of-00001.parquet")
    test_df = pd.read_parquet(f"{data_dir}{height}/test-00000-of-00001.parquet")
    
    # 合并三个数据集
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # 如果时间戳是数值，转为datetime格式
    if 'Date & Time Stamp' in df.columns:
        df['Date & Time Stamp'] = pd.to_datetime(df['Date & Time Stamp'])
    
    # 按时间排序
    df = df.sort_values('Date & Time Stamp').reset_index(drop=True)
    
    # 存储
    dfs[height] = df
    print(f"  {height} 数据加载完成，共 {len(df)} 条记录")

# 合并三个高度的数据
print("\n正在合并三个高度的数据...")
df_merged = dfs['10m'].merge(dfs['50m'], on='Date & Time Stamp', suffixes=('_10m', '_50m'))
df_merged = df_merged.merge(dfs['100m'], on='Date & Time Stamp')

print(f"\n合并完成，总数据量: {len(df_merged)} 条")
print(f"列名: {df_merged.columns.tolist()}")

# ============================================
# 第二部分：数据探索
# ============================================
print("\n" + "="*50)
print("第二步：数据探索")
print("="*50)

print("\n数据基本信息:")
print(df_merged.info())

print("\n数据统计描述:")
print(df_merged.describe())

# 检查缺失值
print("\n缺失值统计:")
print(df_merged.isnull().sum())

# ============================================
# 第三部分：数据预处理
# ============================================
print("\n" + "="*50)
print("第三步：数据预处理")
print("="*50)

# 3.1 处理缺失值
print("处理缺失值...")
df_merged = df_merged.interpolate(method='linear', limit_direction='both')
df_merged = df_merged.dropna()
print(f"处理后数据量: {len(df_merged)}")

# 3.2 处理异常值
print("\n跳过异常值处理...")
# numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
# initial_len = len(df_merged)

# for col in numeric_cols:
#     mean = df_merged[col].mean()
#     std = df_merged[col].std()
#     df_merged = df_merged[(df_merged[col] > mean - 3*std) & (df_merged[col] < mean + 3*std)]

# print(f"异常值处理后数据量: {len(df_merged)} (删除了 {initial_len - len(df_merged)} 条)")
# ============================================
# 第四部分：特征工程
# ============================================
print("\n" + "="*50)
print("第四步：特征工程")
print("="*50)

# 提取时间特征
df_merged['hour'] = df_merged['Date & Time Stamp'].dt.hour
df_merged['dayofweek'] = df_merged['Date & Time Stamp'].dt.dayofweek
df_merged['month'] = df_merged['Date & Time Stamp'].dt.month
df_merged['is_weekend'] = df_merged['dayofweek'].isin([5, 6]).astype(int)

# 目标变量（风速）
target_col = 'SpeedAvg'
print(f"目标变量: {target_col}")

# 构造滞后特征
print("构造滞后特征...")
for lag in [1, 3, 6, 12]:
    df_merged[f'lag_{lag}h'] = df_merged[target_col].shift(lag)

# 构造滑动窗口统计特征
for window in [3, 6]:
    df_merged[f'rolling_{window}h_mean'] = df_merged[target_col].rolling(window).mean()
    df_merged[f'rolling_{window}h_std'] = df_merged[target_col].rolling(window).std()

# 删除含有NaN的行
df_merged = df_merged.dropna()
print(f"特征工程后数据量: {len(df_merged)}")

# 定义特征列
feature_cols = ['hour', 'dayofweek', 'month', 'is_weekend', 
                'lag_1h', 'lag_3h', 'lag_6h', 'lag_12h',
                'rolling_3h_mean', 'rolling_6h_mean',
                'rolling_3h_std', 'rolling_6h_std']

# 添加气象特征
weather_cols = ['TemperatureAvg', 'TemperatureMax', 'PressureAvg', 'PressureMax', 'HumidtyAvg', 'HumityMax']
feature_cols.extend(weather_cols)

print(f"使用的特征: {feature_cols}")

# ============================================
# 第五部分：划分数据集
# ============================================
print("\n" + "="*50)
print("第五步：划分数据集 (7:2:1)")
print("="*50)

total_len = len(df_merged)
train_end = int(0.7 * total_len)
val_end = int(0.9 * total_len)

train_df = df_merged[:train_end]
val_df = df_merged[train_end:val_end]
test_df = df_merged[val_end:]

print(f"训练集: {len(train_df)} 条")
print(f"验证集: {len(val_df)} 条")
print(f"测试集: {len(test_df)} 条")

# 分离X和y
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_val = val_df[feature_cols].values
y_val = val_df[target_col].values
X_test = test_df[feature_cols].values
y_test = test_df[target_col].values

# ============================================
# 第六部分：模型训练和评估
# ============================================
print("\n" + "="*50)
print("第六步：训练模型")
print("="*50)

# 6.1 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

results = {}

# ---------- 模型1: 线性回归 ----------
print("\n1. 训练线性回归模型...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

results['Linear Regression'] = {'MSE': mse_lr, 'RMSE': rmse_lr, 'MAE': mae_lr, 'R2': r2_lr}
print(f"  MSE: {mse_lr:.4f}, RMSE: {rmse_lr:.4f}, MAE: {mae_lr:.4f}, R2: {r2_lr:.4f}")

# ---------- 模型2: LSTM ----------
print("\n2. 训练LSTM模型...")

def create_sequences(data, target, seq_len=24):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(data)):
        X_seq.append(data[i-seq_len:i])
        y_seq.append(target[i])
    return np.array(X_seq), np.array(y_seq)

seq_len = 24
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_len)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, seq_len)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_len)

print(f"  序列数据形状 - 训练: {X_train_seq.shape}, 测试: {X_test_seq.shape}")

X_train_t = torch.FloatTensor(X_train_seq)
y_train_t = torch.FloatTensor(y_train_seq).unsqueeze(1)
X_val_t = torch.FloatTensor(X_val_seq)
y_val_t = torch.FloatTensor(y_val_seq).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test_seq)
y_test_t = torch.FloatTensor(y_test_seq).unsqueeze(1)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        return self.fc(lstm_out)

input_size = X_train_seq.shape[2]
lstm_model = LSTMModel(input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

batch_size = 64
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("  训练中...")
epochs = 50
for epoch in range(epochs):
    lstm_model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = lstm_model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")

lstm_model.eval()
with torch.no_grad():
    y_pred_lstm = lstm_model(X_test_t).numpy().flatten()

mse_lstm = mean_squared_error(y_test_seq, y_pred_lstm)
rmse_lstm = np.sqrt(mse_lstm)
mae_lstm = mean_absolute_error(y_test_seq, y_pred_lstm)
r2_lstm = r2_score(y_test_seq, y_pred_lstm)

results['LSTM'] = {'MSE': mse_lstm, 'RMSE': rmse_lstm, 'MAE': mae_lstm, 'R2': r2_lstm}
print(f"  MSE: {mse_lstm:.4f}, RMSE: {rmse_lstm:.4f}, MAE: {mae_lstm:.4f}, R2: {r2_lstm:.4f}")

# ---------- 模型3: Transformer ----------
print("\n3. 训练Transformer模型...")

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).float().unsqueeze(0).unsqueeze(-1)
        pos_encoding = self.pos_encoder(positions)
        x = self.input_proj(x) + pos_encoding
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

transformer_model = TransformerModel(input_size)
optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)

print("  训练中...")
for epoch in range(epochs):
    transformer_model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = transformer_model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")

transformer_model.eval()
with torch.no_grad():
    y_pred_trans = transformer_model(X_test_t).numpy().flatten()

mse_trans = mean_squared_error(y_test_seq, y_pred_trans)
rmse_trans = np.sqrt(mse_trans)
mae_trans = mean_absolute_error(y_test_seq, y_pred_trans)
r2_trans = r2_score(y_test_seq, y_pred_trans)

results['Transformer'] = {'MSE': mse_trans, 'RMSE': rmse_trans, 'MAE': mae_trans, 'R2': r2_trans}
print(f"  MSE: {mse_trans:.4f}, RMSE: {rmse_trans:.4f}, MAE: {mae_trans:.4f}, R2: {r2_trans:.4f}")

# ============================================
# 第七部分：结果对比
# ============================================
print("\n" + "="*50)
print("第七步：结果对比")
print("="*50)

results_df = pd.DataFrame(results).T
print("\n模型性能对比:")
print(results_df)

# ============================================
# 第八部分：可视化
# ============================================
print("\n" + "="*50)
print("第八步：生成可视化图表")
print("="*50)

try:
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 图1: 风速分布
ax1 = axes[0, 0]
ax1.hist(df_merged[target_col], bins=50, edgecolor='black', alpha=0.7)
ax1.set_title('风速分布直方图')
ax1.set_xlabel('风速 (m/s)')
ax1.set_ylabel('频数')

# 图2: 特征相关性热力图
ax2 = axes[0, 1]
corr_cols = feature_cols + [target_col]
corr_matrix = df_merged[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            ax=ax2, cbar_kws={'label': '相关系数'})
ax2.set_title('特征相关性热力图')

# 图3: 预测结果对比
ax3 = axes[1, 0]
test_len = min(200, len(y_test_seq))
x_axis = range(test_len)
ax3.plot(x_axis, y_test_seq[:test_len], label='真实值', color='black', linewidth=2)
ax3.plot(x_axis, y_pred_lr[:test_len], label='线性回归', color='blue', alpha=0.7)
ax3.plot(x_axis, y_pred_lstm[:test_len], label='LSTM', color='green', alpha=0.7)
ax3.plot(x_axis, y_pred_trans[:test_len], label='Transformer', color='red', alpha=0.7)
ax3.set_title('预测结果对比')
ax3.set_xlabel('时间')
ax3.set_ylabel('风速 (m/s)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4: 模型性能对比柱状图
ax4 = axes[1, 1]
metrics = ['MSE', 'RMSE', 'MAE']
x = np.arange(len(metrics))
width = 0.25
for i, (model_name, model_results) in enumerate(results.items()):
    values = [model_results[m] for m in metrics]
    ax4.bar(x + i*width, values, width, label=model_name)
ax4.set_title('模型性能对比')
ax4.set_xticks(x + width)
ax4.set_xticklabels(metrics)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
print("图片已保存: predictions.png")

# 单独保存分布图
plt.figure(figsize=(10, 6))
plt.hist(df_merged[target_col], bins=50, edgecolor='black', alpha=0.7)
plt.title('风速分布直方图')
plt.xlabel('风速 (m/s)')
plt.ylabel('频数')
plt.savefig('distribution.png', dpi=300, bbox_inches='tight')
print("图片已保存: distribution.png")

# 单独保存相关性热力图
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            cbar_kws={'label': '相关系数'})
plt.title('特征相关性热力图')
plt.savefig('correlation.png', dpi=300, bbox_inches='tight')
print("图片已保存: correlation.png")

print("\n" + "="*50)
print("所有任务完成！")
print("="*50)