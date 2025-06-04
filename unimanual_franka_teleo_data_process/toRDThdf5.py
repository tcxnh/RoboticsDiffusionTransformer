import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm

def convert_hdf5_format(source_file, target_file):
    """
    将原始HDF5格式转换为目标训练格式
    
    参数:
        source_file: 原始HDF5文件路径
        target_file: 转换后保存的目标文件路径
    """
    with h5py.File(source_file, 'r') as src, h5py.File(target_file, 'w') as dst:
        # 1. 转换动作数据
        # 假设action_joint是关节动作，action_ee是末端执行器动作
        if 'action_joint' in src:
            dst.create_dataset('action', data=src['action_joint'][:])
        
        # 2. 转换观察数据
        obs_group = dst.create_group('observations')
        
        # 2.1 转换关节状态
        if 'joint' in src:
            obs_group.create_dataset('qpos', data=src['joint'][:])
        
        # 2.2 转换图像数据
        img_group = obs_group.create_group('images')
        
        # 将color_static转换为cam_high
        if 'color_static' in src:
            static_imgs = src['color_static']
            # 将图像编码为JPEG字节流存储
            encoded_imgs = []
            for i in tqdm(range(static_imgs.shape[0]), desc="编码静态相机图像"):
                img = static_imgs[i]
                if isinstance(img, h5py.Dataset):
                    img = img[:]  # 如果是数据集，获取numpy数组
                _, img_encoded = cv2.imencode('.jpg', img)
                encoded_imgs.append(img_encoded.tobytes())
            img_group.create_dataset('cam_high', data=np.array(encoded_imgs))
        
        # 将color_gripper转换为cam_right_wrist
        if 'color_gripper' in src:
            gripper_imgs = src['color_gripper']
            encoded_imgs = []
            for i in tqdm(range(gripper_imgs.shape[0]), desc="编码夹爪相机图像"):
                img = gripper_imgs[i]
                if isinstance(img, h5py.Dataset):
                    img = img[:]
                _, img_encoded = cv2.imencode('.jpg', img)
                encoded_imgs.append(img_encoded.tobytes())
            img_group.create_dataset('cam_right_wrist', data=np.array(encoded_imgs))
        
        # 3. 添加其他必要的数据组(如果没有则创建空组)
        if 'ee' in src:
            obs_group.create_dataset('ee_pos', data=src['ee'][:])
        
        if 'gripper' in src:
            obs_group.create_dataset('gripper_state', data=src['gripper'][:])

def batch_convert_hdf5(source_dir, target_dir):
    """
    批量转换目录中的所有HDF5文件
    
    参数:
        source_dir: 原始HDF5文件目录
        target_dir: 转换后文件保存目录
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for filename in tqdm(os.listdir(source_dir), desc="处理文件中"):
        if filename.endswith('.hdf5'):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            try:
                convert_hdf5_format(source_path, target_path)
                print(f"成功转换: {filename}")
            except Exception as e:
                print(f"转换失败 {filename}: {str(e)}")

if __name__ == "__main__":
    # 使用示例
    SOURCE_DIR = "/Users/yunzhu/Desktop/teleo_data/pick_up_banana"  # 原始数据目录
    TARGET_DIR = "/Users/yunzhu/Desktop/teleo_data/pick_up_banana_rdt"  # 转换后数据目录
    
    print("开始批量转换HDF5文件格式...")
    batch_convert_hdf5(SOURCE_DIR, TARGET_DIR)
    print("转换完成!")