from utils.ShapeNet import ShapeNet
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# datasetName = 'dataset'
datasetName = 'testset'

# 加载数据集
dataset = ShapeNet('datasets/'+datasetName+'_voxels.tar')
# test_dataset = ShapeNet('datasets/test_dataset_voxels.tar')
# 基础检查
print(f"数据集长度: {len(dataset)}")
sample = dataset[np.random.randint(0, len(dataset))]
print(f"单个样本形状: {sample.shape}")
print(f"数据类型: {sample.dtype}")
print(f"值范围: Min={sample.min()}, Max={sample.max()}")
print(f"非零体素占比: {np.mean(sample > 0.5) * 100:.2f}%")

# 可视化检查
def plot_voxel(voxel, threshold=0.5):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    voxel = voxel.squeeze() > threshold
    ax.voxels(voxel, edgecolor='k', facecolors='red', alpha=0.5)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)
    ax.set_zlim(0, 32)

    plt.title("3D Voxel Visualization")
    plt.show()

plot_voxel(sample)



# 测试数据加载器
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for i, batch in enumerate(loader):
    print(f"批次{i} - 形状: {batch.shape} 范围: [{batch.min()}, {batch.max()}]")
    
    # 检查CUDA转换
    try:
        cuda_batch = batch.to('cuda')
        print("CUDA转换成功")
    except Exception as e:
        print(f"CUDA转换失败: {str(e)}")
    
    if i == 2:  # 只检查前3个批次
        break