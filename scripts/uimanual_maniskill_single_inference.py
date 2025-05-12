#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
单臂推理测试代码 - 使用固定关节位置而非实时数据
修复OpenCV数据类型错误的RealSense相机处理
"""

import argparse
import time
import yaml
import os
from collections import deque
from datetime import datetime

import numpy as np
import torch
from PIL import Image as PImage
import cv2

# 添加PyRealSense2导入
import pyrealsense2 as rs
from scripts.maniskill_model_no_lan_cal import create_model, RoboticDiffusionTransformerModel

# 创建保存图像的目录
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

# 单摄像头设置
CAMERA_NAMES = ['cam_high']

observation_window = None
lang_embeddings = None

# 7自由度加夹爪， 虚拟数据用于测试
DEFAULT_JOINT_POSITION = np.array([0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397,0 # 添加第8个值，"right_gripper_open“ degree
])

# Initialize the model
def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config
    
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args.config, 
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )
    # 重置模型
    model.reset()
    
    return model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_config(args):
    config = {
        'episode_len': args.max_publish_step,
        'state_dim': 7,  # 单臂的状态维度
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,
    }
    return config


# 使用PIL保存图像 - 修复版本，避免使用cv2.cvtColor
def save_pil_image(filepath, img_data):
    """使用PIL保存图像，避开OpenCV的类型问题"""
    try:
        if img_data is None:
            print(f"警告: 尝试保存空图像到 {filepath}")
            return False
            
        # 打印输入图像信息以帮助调试
        print(f"保存图像: 类型={type(img_data)}")
        if isinstance(img_data, np.ndarray):
            print(f"  形状={img_data.shape}, 数据类型={img_data.dtype}")
            
        # 尝试转换为PIL图像并保存
        try:
            if isinstance(img_data, np.ndarray):
                # 确保数据类型是uint8
                if img_data.dtype != np.uint8:
                    img_data = img_data.astype(np.uint8)
                    
                if len(img_data.shape) == 3 and img_data.shape[2] == 3:
                    # 直接使用NumPy进行颜色转换，避免使用OpenCV
                    # 假设输入是BGR格式，交换通道 BGR -> RGB
                    rgb_data = img_data[:, :, ::-1]  # 反转颜色通道顺序
                    pil_img = PImage.fromarray(rgb_data)
                else:
                    pil_img = PImage.fromarray(img_data)
            else:
                # 如果已经是PIL图像或其他格式
                pil_img = PImage.fromarray(np.array(img_data))
                
            pil_img.save(filepath)
            print(f"已使用PIL保存图像到: {filepath}")
            return True
            
        except Exception as e:
            print(f"使用PIL保存图像时出错: {e}")
            return False
            
    except Exception as e:
        print(f"保存图像过程中出错: {e}")
        return False


# 初始化RealSense相机 - 仅使用彩色流
def initialize_realsense():
    try:
        # 创建RealSense管道
        pipeline = rs.pipeline()
        config = rs.config()
        
        # 获取可用设备列表
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("错误: 未检测到RealSense设备")
            return None
            
        print(f"检测到 {len(devices)} 个RealSense设备")
        for i, dev in enumerate(devices):
            print(f"  设备 {i}: {dev.get_info(rs.camera_info.name)}, 序列号: {dev.get_info(rs.camera_info.serial_number)}")
        
        # 只配置彩色流，不使用深度流
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 开始流
        try:
            profile = pipeline.start(config)
            print("RealSense管道已启动")
            
            # 打印活动流信息
            for stream in profile.get_streams():
                stream_profile = stream.as_video_stream_profile()
                print(f"  活动流: {stream.stream_name()}, 格式: {stream.format()}, " 
                      f"分辨率: {stream_profile.width()}x{stream_profile.height()}")
                
        except Exception as e:
            print(f"启动RealSense管道时出错: {e}")
            return None
        
        # 预热相机 - 更健壮的方法
        print("正在预热相机...")
        successful_frames = 0
        max_attempts = 60  # 最多尝试60次，以获取30个成功的帧
        attempt = 0
        
        while successful_frames < 30 and attempt < max_attempts:
            try:
                attempt += 1
                frame = pipeline.wait_for_frames(2000)  # 2秒超时，更合理
                color_frame = frame.get_color_frame()
                
                if color_frame:
                    successful_frames += 1
                    if successful_frames % 5 == 0:
                        print(f"  预热进度: {successful_frames}/30 (尝试 {attempt}/{max_attempts})")
                        
                    # 每10帧保存一次测试图像
                    if successful_frames % 10 == 0:
                        try:
                            # 直接从color_frame获取数据
                            img_data = np.asanyarray(color_frame.get_data())
                            timestamp = datetime.now().strftime("%H%M%S%f")
                            test_img_path = os.path.join(DEBUG_DIR, f"warmup_test_{timestamp}.jpg")
                            
                            # 使用PIL保存图像
                            save_pil_image(test_img_path, img_data)
                        except Exception as e:
                            print(f"  保存预热测试图像时出错: {e}")
                else:
                    print(f"  预热尝试 {attempt}: 未获取到彩色帧")
            except Exception as e:
                print(f"  预热尝试 {attempt} 出错: {e}")
                time.sleep(0.5)  # 短暂暂停，让相机恢复
        
        if successful_frames < 10:  # 如果成功获取的帧少于10个，则认为预热失败
            print(f"相机预热不成功，只获取了 {successful_frames}/30 个帧")
            pipeline.stop()
            return None
            
        print(f"相机预热完成，成功获取了 {successful_frames} 个帧")
        return pipeline
        
    except Exception as e:
        print(f"初始化RealSense相机时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


# 获取相机图像
def get_camera_image(pipeline):
    try:
        # 尝试多次获取帧
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # 等待帧，设置超时
                frames = pipeline.wait_for_frames(2000)  # 2秒超时
                
                # 获取彩色帧
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    print(f"警告: 第 {attempt+1}/{max_attempts} 次尝试未收到颜色帧")
                    time.sleep(0.2)  # 短暂暂停
                    continue
                    
                # 转换为NumPy数组 - 确保类型正确
                color_image = np.asanyarray(color_frame.get_data())
                
                # 验证图像
                if color_image is None or color_image.size == 0:
                    print(f"警告: 第 {attempt+1}/{max_attempts} 次尝试颜色图像为空或大小为0")
                    continue
                    
                if len(color_image.shape) != 3:
                    print(f"警告: 第 {attempt+1}/{max_attempts} 次尝试颜色图像维度不正确, 形状: {color_image.shape}")
                    continue
                
                # 确保数据类型为uint8
                if color_image.dtype != np.uint8:
                    color_image = color_image.astype(np.uint8)
                
                # 打印数据类型和形状以进行诊断
                print(f"获取的图像: 形状={color_image.shape}, 类型={color_image.dtype}, 数据范围=[{np.min(color_image)}, {np.max(color_image)}]")
                
                # 保存原始图像 - 使用PIL而非OpenCV
                timestamp = datetime.now().strftime("%H%M%S%f")
                raw_img_path = os.path.join(DEBUG_DIR, f"raw_image_{timestamp}.jpg")
                
                if save_pil_image(raw_img_path, color_image):
                    print(f"已保存原始图像到: {raw_img_path}")
                    return color_image
                else:
                    print(f"警告: 第 {attempt+1}/{max_attempts} 次尝试保存原始图像失败")
                    
            except Exception as e:
                print(f"第 {attempt+1}/{max_attempts} 次尝试获取相机图像时出错: {e}")
                time.sleep(0.5)  # 短暂暂停
                
        print(f"在 {max_attempts} 次尝试后仍未能获取有效的相机图像")
        
        # 创建一个测试图像作为备用方案 - 使用numpy直接创建，避免OpenCV
        print("创建测试图像作为备用方案...")
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # 添加绿色边框 - 不使用OpenCV函数
        test_image[100:103, 100:540] = [0, 255, 0]  # 上边框
        test_image[380:383, 100:540] = [0, 255, 0]  # 下边框
        test_image[100:380, 100:103] = [0, 255, 0]  # 左边框
        test_image[100:380, 537:540] = [0, 255, 0]  # 右边框
        
        # 在中间添加一些文字的简单图案
        test_image[220:260, 220:420] = [255, 255, 255]  # 白色文字区域
        
        timestamp = datetime.now().strftime("%H%M%S%f")
        test_img_path = os.path.join(DEBUG_DIR, f"test_image_{timestamp}.jpg")
        
        if save_pil_image(test_img_path, test_image):
            print(f"已保存测试图像到: {test_img_path}")
            return test_image
        else:
            print("测试图像也无法保存，创建最小测试图像")
            # 创建最小的有效测试图像
            return np.ones((480, 640, 3), dtype=np.uint8) * 128  # 灰色图像
            
    except Exception as e:
        print(f"获取相机图像时出错: {e}")
        import traceback
        traceback.print_exc()
        # 返回一个最小的有效测试图像作为最后的备用方案
        return np.ones((480, 640, 3), dtype=np.uint8) * 128  # 灰色图像


# 使用PIL进行JPEG压缩，避免OpenCV类型问题 - 修复版本
def jpeg_mapping(img):
    try:
        # 检查输入
        if img is None:
            print("警告: jpeg_mapping收到空图像")
            return None
            
        if not isinstance(img, np.ndarray):
            print(f"警告: jpeg_mapping收到非numpy数组类型: {type(img)}")
            try:
                img = np.array(img)
                print(f"已转换为numpy数组, 形状: {img.shape}")
            except:
                print("无法转换为numpy数组")
                return None
        
        # 保存JPEG压缩前的图像
        timestamp = datetime.now().strftime("%H%M%S%f")
        before_jpg_path = os.path.join(DEBUG_DIR, f"before_jpg_{timestamp}.jpg")
        save_pil_image(before_jpg_path, img)
        
        # 使用PIL进行JPEG压缩 - 确保类型转换正确
        try:
            # 确保img是uint8类型
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
                
            # 如果是BGR格式，转换为RGB - 使用NumPy操作而非OpenCV
            if len(img.shape) == 3 and img.shape[2] == 3:
                rgb_img = img[:, :, ::-1]  # BGR -> RGB 通过反转通道顺序
            else:
                rgb_img = img
                
            # 转换为PIL图像
            pil_img = PImage.fromarray(rgb_img)
            
            # 临时保存为JPEG
            jpg_buffer = os.path.join(DEBUG_DIR, f"temp_{timestamp}.jpg")
            pil_img.save(jpg_buffer, format='JPEG', quality=95)
            
            # 重新读取JPEG
            decoded_pil_img = PImage.open(jpg_buffer)
            decoded_array = np.array(decoded_pil_img)
            
            # 转回BGR格式 - 使用NumPy操作而非OpenCV
            if len(decoded_array.shape) == 3 and decoded_array.shape[2] == 3:
                decoded_img = decoded_array[:, :, ::-1]  # RGB -> BGR 通过反转通道顺序
            else:
                decoded_img = decoded_array
            
            # 保存压缩后的图像
            after_jpg_path = os.path.join(DEBUG_DIR, f"after_jpg_{timestamp}.jpg")
            save_pil_image(after_jpg_path, decoded_img)
            print(f"已进行JPEG压缩，保存到: {after_jpg_path}")
            
            return decoded_img
            
        except Exception as e:
            print(f"JPEG压缩过程中出错: {e}")
            print("返回原始图像")
            return img
            
    except Exception as e:
        print(f"JPEG映射过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return img  # 返回原始图像


# 更新观察窗口
def update_observation_window(config, pipeline):
    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)
    
        # 添加第一个空图像
        observation_window.append(
            {
                'qpos': None,
                'images': {config["camera_names"][0]: None},
            }
        )
    
    # 获取相机图像
    print("正在获取相机图像...")
    img_front = get_camera_image(pipeline)
    if img_front is None:
        print("错误: 无法获取相机图像")
        return False
        
    print("正在进行JPEG压缩处理...")
    img_front = jpeg_mapping(img_front)
    if img_front is None:
        print("错误: JPEG压缩处理失败")
        return False
    
    # 使用固定的关节位置来替代实时数据
    qpos = torch.from_numpy(DEFAULT_JOINT_POSITION).float().cuda()
    
    observation_window.append(
        {
            'qpos': qpos,
            'images': {config["camera_names"][0]: img_front},
        }
    )
    
    print("成功更新观察窗口")
    return True


# RDT 推理 - 修复版本
def inference_fn(config, policy, t, save_images=True):
    global observation_window
    global lang_embeddings
    
    time1 = time.time()     

    # 准备PIL图像列表，按照模型期望的格式整理
    images_list = []
    for t_idx in range(2):  # 2个时间步
        if t_idx == 0 and observation_window[t_idx]['images'][config['camera_names'][0]] is None:
            # 第一帧不可用时
            images_list.append(None)
        else:
            # 将BGR图像转换为RGB格式的PIL图像
            cv_img = observation_window[t_idx]['images'][config['camera_names'][0]]
            if cv_img is not None:
                try:
                    # 确保是numpy数组
                    if not isinstance(cv_img, np.ndarray):
                        cv_img = np.array(cv_img)
                    
                    # 确保是uint8类型
                    if cv_img.dtype != np.uint8:
                        cv_img = cv_img.astype(np.uint8)
                    
                    # 转RGB格式 - 使用NumPy操作而非OpenCV
                    rgb_img = cv_img[:, :, ::-1]  # BGR -> RGB 通过反转通道顺序
                    
                    # 创建PIL图像
                    pil_img = PImage.fromarray(rgb_img)
                    
                    if save_images:
                        pil_path = os.path.join(DEBUG_DIR, f'pil_input_{t_idx}.jpg')
                        pil_img.save(pil_path)
                        print(f"已保存输入PIL图像 {t_idx} 到: {pil_path}")
                    
                    images_list.append(pil_img)
                except Exception as e:
                    print(f"转换图像 {t_idx} 为PIL格式时出错: {e}")
                    # 创建一个空白的PIL图像作为备用
                    backup_img = PImage.new('RGB', (640, 480), color=(128, 128, 128))
                    images_list.append(backup_img)
            else:
                # 创建一个空白的PIL图像作为备用
                backup_img = PImage.new('RGB', (640, 480), color=(128, 128, 128))
                images_list.append(backup_img)
        
        # 添加两个None，以匹配模型期望的格式
        images_list.append(None)
        images_list.append(None)
    
    # 获取关节状态
    proprio = observation_window[-1]['qpos']
    # 扩展维度以匹配模型输入要求 [1, 7]
    proprio = proprio.unsqueeze(0)
    
    print("开始模型推理...")
    # 生成动作序列
    try:
        actions = policy.step(
            proprio=proprio,
            images=images_list,
            text_embeds=lang_embeddings 
        ).squeeze(0).cpu().numpy()
        
        print(f"模型推理时间: {time.time() - time1} 秒")
        return actions
        
    except Exception as e:
        print(f"模型推理出错: {e}")
        import traceback
        traceback.print_exc()
        return None


# 主测试函数
def test_inference(args):
    global lang_embeddings
    
    # 初始化配置
    config = get_config(args)
    
    # 初始化相机
    pipeline = initialize_realsense()
    if pipeline is None:
        print("无法初始化相机，测试退出")
        return
    
    try:
        # 加载模型
        print("正在加载模型...")
        policy = make_policy(args)
        
        # 加载语言嵌入
        print("正在加载语言嵌入...")
        try:
            lang_dict = torch.load(args.lang_embeddings_path)
            print(f"运行指令: \"{lang_dict['instruction']}\" 来自 \"{lang_dict['name']}\"")
            lang_embeddings = lang_dict["embeddings"]
        except Exception as e:
            print(f"加载语言嵌入时出错: {e}")
            return
        
        # 等待初始化完成
        input("相机已初始化。按Enter键开始推理测试...")
        
        # 更新观察窗口 - 获取两帧图像
        print("获取第一帧相机图像...")
        success = update_observation_window(config, pipeline)
        if not success:
            print("无法获取相机图像，测试退出")
            return
            
        print("等待获取第二帧...")
        time.sleep(0.5)  # 等待获取第二帧
        
        success = update_observation_window(config, pipeline)
        if not success:
            print("无法获取第二帧相机图像，测试退出")
            return
        
        # 执行推理
        print("执行模型推理...")
        with torch.inference_mode():
            actions = inference_fn(config, policy, 0, save_images=True)
        
        if actions is None:
            print("模型推理失败，未生成动作")
            return
            
        # 打印结果
        print("\n推理结果 - 生成的动作序列:")
        print(f"动作形状: {actions.shape}")
        print("前5个时间步的动作:")
        for i in range(min(5, actions.shape[0])):
            print(f"步骤 {i}: {actions[i]}")
            
        # 保存完整结果
        np.save("inference_actions.npy", actions)
        print("\n完整动作序列已保存到 'inference_actions.npy'")
        
    finally:
        # 清理资源
        pipeline.stop()
        print("测试完成，相机资源已释放")
        print(f"所有调试图像都保存在 {DEBUG_DIR} 目录中")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int, 
                        help='Maximum number of action publishing steps', default=10000, required=False)
    parser.add_argument('--seed', action='store', type=int, 
                        help='Random seed', default=None, required=False)
    
    parser.add_argument('--ctrl_freq', action='store', type=int, 
                        help='The control frequency of the robot',
                        default=25, required=False)
    
    parser.add_argument('--chunk_size', action='store', type=int, 
                        help='Action chunk size',
                        default=64, required=False)
    
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", 
                        help='Path to the config file')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='Name or path to the pretrained model')
    
    parser.add_argument('--lang_embeddings_path', type=str, required=True, 
                        help='Path to the pre-encoded language instruction embeddings')
    
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
        
    # 执行测试推理
    test_inference(args)


if __name__ == '__main__':
    main()