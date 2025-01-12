# Readme #
##文件架构说明##
├── .idea/                 # 开发工具（PyCharm）生成的配置文件夹
├── .idea/                 # 开发工具（如PyCharm）生成的配置文件夹
├── output汇总/            # 输出结果文件的汇总目录（如轨迹文件、实验结果等）
├── .gitattributes         # Git属性配置文件
├── Readme.md              # 项目说明文档，介绍项目用途、使用方法等
├── requirements.txt       # 项目依赖的Python库列表
├── trajectory_planning.py # 主程序代码
├── trajectory_planning.py # 主程序代码，用于喷涂轨迹规划
├── 任务1-左前门模型.obj    # 任务1的3D模型文件（左前门部件）
├── 任务2-左前翼子板模型.obj # 任务2的3D模型文件（左前翼子板部件）
├── 任务3-前盖模型.obj      # 任务3的3D模型文件（前盖部件）

## intrudctions： ##

先创建虚拟环境，这里不赘述

1. Install the requirements:

   ```
   pip install -r requirements.txt
   ```

2.  torch库直接用pip可能会因为网络问题报错，建议使用清华源

   ```
   pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple 
   ```

3. 终端命令：用 "任务1-左前门模型.obj"进行预测，输出为“trajectory.json”为例运行程序

   `<input_model_file>`：输入的 3D 模型文件路径（如 `任务1-左前门模型.obj`）。

   `<output_json_file>`：输出的 JSON 文件路径（如 `任务一trajectory.json`）。

   `--spacing <spacing>`：叠枪距离，单位为毫米（如 `80` 或 `100`）。

   `--epochs <epochs>`：训练的迭代次数，默认为 1000 次。

   `--lr <learning_rate>`：学习率，默认为 0.001。

   例如：

   ```
   python trajectory_planning.py 任务1-左前门模型.obj 任务一trajectory.json --spacing 80 --epochs 1000 --lr 0.001
   ```

代码正常运行结果呈现：

![image-20250113013419070](C:\Users\LENOVO\PycharmProjects\trajectory_planning\output汇总\image-20250113013419070.png)

## 基于多目标规划模型的喷涂轨迹规划神经网络购建说明 ##

在本项目中，使用典型的多层感知机，将车辆部件的 3D 表面点作为输入，并输出喷涂轨迹点的位置。通过训练，神经网络能够自动学习如何将输入的轨迹点位置映射到喷涂轨迹，进而优化喷涂路径的规划。

### 网络参数

在本模型中，网络包含了三个全连接层（`Linear`）：

#### 1. 第一层（`fc1`）

- **输入维度**：3（输入的是 3D 坐标点）。
- **输出维度**：128
- **激活函数**：ReLU（Rectified Linear Unit）。增加非线性。

#### 2. 第二层（`fc2`）

- **输入维度**：128（上一层的输出维度）。
- **输出维度**：128（这一层的神经元数量与上一层相同）。
- **激活函数**：ReLU

#### 3. 第三层（`fc3`）

- **输入维度**：128（上一层的输出维度）。
- **输出维度**：3（最后一层输出 3D 坐标）。
- **激活函数**：线性输出

### 损失函数

使用均方误差（MSE）作为损失函数来衡量预测轨迹点与真实轨迹点之间的差距。MSE 可以帮助网络更好地拟合目标轨迹点。

### 优化器

常用的优化算法是 Adam，基于梯度下降的优化算法。