import json
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def load_3d_model(file_path):
    """加载3D模型，返回表面点"""
    mesh = trimesh.load(file_path)
    return mesh.vertices, mesh.faces


def load_trajectory(file_path):
    """加载轨迹文件，返回轨迹点"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    points = [point['p'] for point in data['traj_surface']]
    return points


def visualize_model_and_trajectory(vertices, trajectory_points):
    """可视化3D模型和轨迹"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D模型表面点
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, c='lightgray', label='3D Model Surface')

    # 绘制轨迹点
    trajectory_points = np.array(trajectory_points)
    ax.scatter(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], c='red',
               label='Trajectory Points', s=20)

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Model with Trajectory Points')
    ax.legend()
    plt.show()


def main():
    # 输入文件路径
    model_file = '任务1-左前门模型.obj'  # 替换为您的模型文件路径
    trajectory_file = 'output汇总/任务一trajectory.json'  # 替换为您的轨迹文件路径

    # 加载数据
    vertices, _ = load_3d_model(model_file)
    trajectory_points = load_trajectory(trajectory_file)

    # 可视化
    visualize_model_and_trajectory(vertices, trajectory_points)


if __name__ == '__main__':
    main()

