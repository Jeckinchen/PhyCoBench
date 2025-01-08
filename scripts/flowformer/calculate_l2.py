import numpy as np

def calculate_l2_loss(file1, file2):
    # 读取npy文件
    data1 = np.load(file1)
    data2 = np.load(file2)
    
    # 判断形状是否相同
    if data1.shape != data2.shape:
        print("形状不同:", data1.shape, data2.shape)
        return None
    
    # 计算损失
    l2_loss = np.sum((data1 - data2) ** 2)
    l1_loss = np.sum(np.abs(data1 - data2))
    return l1_loss

# 示例用法
file1 = '/mnt/merchant/yongfan/code/DynamiCrafter_2/DynamiCrafter/results/flow_vis_test/video/fall_cardboard_09_foam_01_Camera_1/rgb/oneone/optical_flows.npy'
file2 = '/mnt/merchant/yongfan/code/DynamiCrafter_2/DynamiCrafter/results/flow_vis_test/video/fall_cardboard_09_foam_01_Camera_1_sample0/rgb/oneone/optical_flows.npy'
#file2 = '/mnt/merchant/yongfan/code/DynamiCrafter_2/DynamiCrafter/results/flow_vis_test/video/fall_cardboard_09_foam_01_Camera_1_sample0/rgb/oneone/optical_flows.npy'
loss = calculate_l2_loss(file1, file2)

if loss is not None:
    print("L1损失:", loss)
