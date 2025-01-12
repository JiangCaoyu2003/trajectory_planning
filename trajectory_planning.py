import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import trimesh
import json
from torch.utils.data import Dataset, DataLoader


# 定义神经网络模型
class TrajectoryNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TrajectoryNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 加载.obj文件并获取网格数据
def load_obj(file_path):
    mesh = trimesh.load_mesh(file_path)
    return mesh


# 生成喷涂轨迹点的示例
def generate_trajectory_points(mesh, spacing):
    # 假设每个轨迹点的位置都在表面上均匀分布（这里只是一个示例）
    points = []
    for face in mesh.faces:
        center = mesh.vertices[face].mean(axis=0)
        points.append({"p": center.tolist()})
    return points


# 计算损失函数（包括均匀性、平滑度等）
def total_loss(predicted_trajectory, targets, positions, target_positions):
    # 在此处可以自定义多种损失项，例如均匀性损失、平滑度损失等
    loss = torch.mean((predicted_trajectory - targets) ** 2)
    return loss


# 保存轨迹到JSON文件
def save_trajectory_to_json(trajectory_points, output_file):
    with open(output_file, 'w') as f:
        json.dump({"traj_surface": trajectory_points}, f, indent=4)


# 定义Dataset类（如果使用DataLoader）
class TrajectoryDataset(Dataset):
    def __init__(self, trajectory_points):
        self.trajectory_points = trajectory_points

    def __len__(self):
        return len(self.trajectory_points)

    def __getitem__(self, idx):
        point = self.trajectory_points[idx]
        input_point = torch.tensor(point["p"], dtype=torch.float32)  # 输入
        target_point = torch.tensor(point["p"], dtype=torch.float32)  # 输出（假设预测位置为目标位置）
        return input_point, target_point


# 训练函数
def train(model, optimizer, positions, trajectory_points, epochs=10):
    """训练模型并显示学习率和迭代次数"""
    for epoch in range(epochs):
        model.train()

        inputs = positions
        targets = torch.tensor([point["p"] for point in trajectory_points], dtype=torch.float32)

        optimizer.zero_grad()

        # 前向传播
        predicted_trajectory = model(inputs)

        # 计算损失
        loss = total_loss(predicted_trajectory, targets, predicted_trajectory, targets)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 获取当前学习率
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        # 每10个epoch打印一次学习率和当前的迭代次数
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Learning Rate: {learning_rate:.6f}")


# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="喷涂轨迹规划")
    parser.add_argument('input_file', type=str, help='3D车部件模型文件路径 (.obj)')
    parser.add_argument('output_file', type=str, help='输出轨迹文件路径 (.json)')
    parser.add_argument('--spacing', type=float, default=80.0, help='叠枪距离 (毫米)')
    parser.add_argument('--epochs', type=int, default=10, help='训练的迭代次数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    args = parser.parse_args()

    # 加载3D模型
    mesh = load_obj(args.input_file)

    # 生成喷涂轨迹点
    trajectory_points = generate_trajectory_points(mesh, args.spacing)

    # 提取轨迹点的位置（'p'字段），并将其转换为数值张量
    positions = [np.array(point["p"]) for point in trajectory_points]
    positions = torch.tensor(positions, dtype=torch.float32)

    # 创建并训练模型
    model = TrajectoryNet(input_dim=3, hidden_dim=128, output_dim=3)  # 修改模型输入输出维度
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练
    train(model, optimizer, positions, trajectory_points, epochs=args.epochs)

    # 使用模型生成最终轨迹（预测的轨迹点）
    predicted_trajectory = model(positions)

    # 将预测的轨迹点（包括喷涂状态等）保存为JSON文件
    for i, point in enumerate(predicted_trajectory):
        trajectory_points[i]["p"] = point.tolist()  # 更新轨迹点位置

    # 保存轨迹到JSON文件
    save_trajectory_to_json(trajectory_points, args.output_file)


if __name__ == "__main__":
    main()
