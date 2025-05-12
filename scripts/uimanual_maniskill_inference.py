#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
单臂推理测试代码 - 使用Franka真实关节状态而非固定数据
集成FrankaIOController以获取机器人真实状态
"""

import argparse
import time
import yaml
import os
import rospy
from collections import deque
from datetime import datetime

import numpy as np
import torch
from PIL import Image as PImage
import cv2

# 添加PyRealSense2导入
import pyrealsense2 as rs
from scripts.maniskill_model_no_lan_cal import create_model, RoboticDiffusionTransformerModel

from scripts.franka_ros_control import FrankaIOController

# 创建保存图像的目录
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

CAMERA_NAMES = ['cam_right_wrist', 'cam_high']

observation_window = None
lang_embeddings = None
franka_controller = None  # 添加Franka控制器实例

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
        'state_dim': 8,  
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


# 初始化RealSense相机 - 支持两个相机
def initialize_realsense():
    try:
        # 创建RealSense上下文
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) < 2:
            print(f"错误: 只检测到 {len(devices)} 个RealSense设备，需要2个")
            return None
            
        print(f"检测到 {len(devices)} 个RealSense设备")
        for i, dev in enumerate(devices):
            print(f"  设备 {i}: {dev.get_info(rs.camera_info.name)}, 序列号: {dev.get_info(rs.camera_info.serial_number)}")
        
        # 创建两个pipeline和配置
        pipeline1 = rs.pipeline()
        config1 = rs.config()
        pipeline2 = rs.pipeline()
        config2 = rs.config()
        
        # 为第一个相机启用彩色流
        serial1 = devices[0].get_info(rs.camera_info.serial_number)
        config1.enable_device(serial1)
        config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 为第二个相机启用彩色流
        serial2 = devices[1].get_info(rs.camera_info.serial_number)
        config2.enable_device(serial2)
        config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 启动两个pipeline
        try:
            print(f"正在启动第一个相机 (序列号: {serial1})...")
            profile1 = pipeline1.start(config1)
            print(f"正在启动第二个相机 (序列号: {serial2})...")
            profile2 = pipeline2.start(config2)
            print("两个RealSense管道已启动")
            
            # 打印活动流信息
            for i, profile in enumerate([profile1, profile2]):
                for stream in profile.get_streams():
                    stream_profile = stream.as_video_stream_profile()
                    print(f"  相机 {i} 活动流: {stream.stream_name()}, 格式: {stream.format()}, " 
                          f"分辨率: {stream_profile.width()}x{stream_profile.height()}")
                    
        except Exception as e:
            print(f"启动RealSense管道时出错: {e}")
            return None
        
        # 预热两个相机
        print("正在预热相机...")
        for i, pipeline in enumerate([pipeline1, pipeline2]):
            print(f"预热相机 {i}...")
            successful_frames = 0
            max_attempts = 60
            attempt = 0
            
            while successful_frames < 30 and attempt < max_attempts:
                try:
                    attempt += 1
                    frame = pipeline.wait_for_frames(2000)
                    color_frame = frame.get_color_frame()
                    
                    if color_frame:
                        successful_frames += 1
                        if successful_frames % 5 == 0:
                            print(f"  相机 {i} 预热进度: {successful_frames}/30 (尝试 {attempt}/{max_attempts})")
                            
                        # 每10帧保存一次测试图像
                        if successful_frames % 10 == 0:
                            try:
                                img_data = np.asanyarray(color_frame.get_data())
                                timestamp = datetime.now().strftime("%H%M%S%f")
                                test_img_path = os.path.join(DEBUG_DIR, f"warmup_test_cam{i}_{timestamp}.jpg")
                                
                                save_pil_image(test_img_path, img_data)
                            except Exception as e:
                                print(f"  保存相机 {i} 预热测试图像时出错: {e}")
                    else:
                        print(f"  相机 {i} 预热尝试 {attempt}: 未获取到彩色帧")
                except Exception as e:
                    print(f"  相机 {i} 预热尝试 {attempt} 出错: {e}")
                    time.sleep(0.5)
        
            if successful_frames < 10:
                print(f"相机 {i} 预热不成功，只获取了 {successful_frames}/30 个帧")
                pipeline1.stop()
                pipeline2.stop()
                return None
                
        print("两个相机预热完成")
        return [pipeline1, pipeline2]  # 返回包含两个pipeline的列表
        
    except Exception as e:
        print(f"初始化RealSense相机时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

# 获取相机图像 - 支持两个相机
def get_camera_images(pipelines):
    """从两个相机获取图像"""
    images = []
    
    for i, pipeline in enumerate(pipelines):
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
                        print(f"警告: 相机 {i} 第 {attempt+1}/{max_attempts} 次尝试未收到颜色帧")
                        time.sleep(0.2)  # 短暂暂停
                        continue
                        
                    # 转换为NumPy数组
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # 验证图像
                    if color_image is None or color_image.size == 0:
                        print(f"警告: 相机 {i} 第 {attempt+1}/{max_attempts} 次尝试颜色图像为空或大小为0")
                        continue
                        
                    if len(color_image.shape) != 3:
                        print(f"警告: 相机 {i} 第 {attempt+1}/{max_attempts} 次尝试颜色图像维度不正确, 形状: {color_image.shape}")
                        continue
                    
                    # 确保数据类型为uint8
                    if color_image.dtype != np.uint8:
                        color_image = color_image.astype(np.uint8)
                    
                    # 打印数据类型和形状以进行诊断
                    print(f"获取的相机 {i} 图像: 形状={color_image.shape}, 类型={color_image.dtype}, 范围=[{np.min(color_image)}, {np.max(color_image)}]")
                    
                    # 保存原始图像
                    timestamp = datetime.now().strftime("%H%M%S%f")
                    raw_img_path = os.path.join(DEBUG_DIR, f"raw_image_cam{i}_{timestamp}.jpg")
                    
                    if save_pil_image(raw_img_path, color_image):
                        print(f"已保存相机 {i} 原始图像到: {raw_img_path}")
                        images.append(color_image)
                        break
                    else:
                        print(f"警告: 相机 {i} 第 {attempt+1}/{max_attempts} 次尝试保存原始图像失败")
                        
                except Exception as e:
                    print(f"相机 {i} 第 {attempt+1}/{max_attempts} 次尝试获取图像时出错: {e}")
                    time.sleep(0.5)  # 短暂暂停
                    
            # 如果经过多次尝试仍未获取图像，使用备用图像
            if len(images) <= i:
                print(f"在 {max_attempts} 次尝试后仍未能获取相机 {i} 的有效图像")
                
                # 创建一个测试图像作为备用方案
                print(f"创建相机 {i} 的测试图像作为备用方案...")
                test_image = np.zeros((480, 640, 3), dtype=np.uint8)
                # 添加标识该相机的彩色边框
                color = [0, 255, 0] if i == 0 else [0, 0, 255]  # 第一个相机绿色，第二个相机蓝色
                test_image[100:103, 100:540] = color  # 上边框
                test_image[380:383, 100:540] = color  # 下边框
                test_image[100:380, 100:103] = color  # 左边框
                test_image[100:380, 537:540] = color  # 右边框
                
                # 在中间添加相机编号
                test_image[220:260, 300:340] = [255, 255, 255]  # 白色文字区域
                
                timestamp = datetime.now().strftime("%H%M%S%f")
                test_img_path = os.path.join(DEBUG_DIR, f"test_image_cam{i}_{timestamp}.jpg")
                
                if save_pil_image(test_img_path, test_image):
                    print(f"已保存相机 {i} 的测试图像到: {test_img_path}")
                    images.append(test_image)
                else:
                    print(f"相机 {i} 的测试图像也无法保存，创建最小测试图像")
                    # 创建最小的有效测试图像
                    images.append(np.ones((480, 640, 3), dtype=np.uint8) * 128)  # 灰色图像
                
        except Exception as e:
            print(f"获取相机 {i} 图像时出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个最小的有效测试图像作为最后的备用方案
            images.append(np.ones((480, 640, 3), dtype=np.uint8) * 128)  # 灰色图像
    
    return images


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


# 初始化Franka控制器
def initialize_franka_controller(rate=10):
    """初始化Franka控制器以获取真实关节数据"""
    try:
        print("正在初始化Franka控制器...")
        controller = FrankaIOController(rate=rate)
        
        # 等待获取第一帧关节状态数据
        if not controller.wait_for_joint_state(timeout=5.0):
            print("错误: 无法获取Franka关节状态！")
            return None
            
        print("成功初始化Franka控制器")
        return controller
    except Exception as e:
        print(f"初始化Franka控制器时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


# 获取Franka关节位置
def get_franka_joint_positions(controller):
    """从Franka控制器获取当前关节位置"""
    try:
        joint_positions = controller.get_current_joint_positions()
        
        if joint_positions is None:
            print("警告: 无法获取Franka关节位置，返回None")
            return None
            
        # 只取前7个关节位置，忽略其他值
        joint_positions = joint_positions[:7]
        
        # 转换为numpy数组
        joint_positions = np.array(joint_positions, dtype=np.float32)
        
        print(f"获取的关节位置: {joint_positions}")
        return joint_positions
        
    except Exception as e:
        print(f"获取Franka关节位置时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


# 更新观察窗口 - 支持两个相机
def update_observation_window(config, pipelines, franka_controller):
    global observation_window
    
    if observation_window is None:
        observation_window = deque(maxlen=2)
    
        # 添加第一个空图像和关节位置
        empty_images = {cam_name: None for cam_name in config["camera_names"]}
        observation_window.append(
            {
                'qpos': None,
                'images': empty_images,
            }
        )
    
    # 获取相机图像
    print("正在获取相机图像...")
    images = get_camera_images(pipelines)
    if len(images) < 2:
        print(f"错误: 只获取到 {len(images)} 个相机图像，需要2个")
        return False
    
    # 处理两个图像
    processed_images = {}
    for i, cam_name in enumerate(config["camera_names"]):
        print(f"正在处理相机 {cam_name} 的图像...")
        img = jpeg_mapping(images[i])
        if img is None:
            print(f"错误: 相机 {cam_name} 的JPEG压缩处理失败")
            return False
        processed_images[cam_name] = img
    
    # 获取真实的Franka关节位置
    print("正在获取Franka关节位置...")
    joint_positions = get_franka_joint_positions(franka_controller)
    
    if joint_positions is None:
        print("错误: 无法获取Franka关节位置，使用零向量代替")
        joint_positions = np.zeros(7, dtype=np.float32)
    
    # 转换为PyTorch张量并移到GPU
    qpos = torch.from_numpy(joint_positions).float().cuda()
    
    # 更新观察窗口
    observation_window.append(
        {
            'qpos': qpos,
            'images': processed_images,
        }
    )
    
    print("成功更新观察窗口")
    return True


# 推理函数 - 添加第三个虚拟相机以匹配双臂代码
def inference_fn(config, policy, t, save_images=True):
    global observation_window
    global lang_embeddings
    
    time1 = time.time()     

    # 准备PIL图像列表，模拟三个相机的配置
    images = []
    
    # 第一帧三个相机 (t-1)
    # 真实相机: camera_high 和 camera_right_wrist
    for cam_name in config['camera_names']:
        if observation_window[-2]['images'][cam_name] is None:
            images.append(PImage.new('RGB', (640, 480), color=(128, 128, 128)))
        else:
            cv_img = observation_window[-2]['images'][cam_name]
            try:
                # 确保是numpy数组
                if not isinstance(cv_img, np.ndarray):
                    cv_img = np.array(cv_img)
                
                # 确保是uint8类型
                if cv_img.dtype != np.uint8:
                    cv_img = cv_img.astype(np.uint8)
                
                # 转RGB格式
                rgb_img = cv_img[:, :, ::-1]  # BGR -> RGB
                
                # 创建PIL图像
                pil_img = PImage.fromarray(rgb_img)
                
                if save_images:
                    pil_path = os.path.join(DEBUG_DIR, f'pil_input_{cam_name}_0.jpg')
                    pil_img.save(pil_path)
                    print(f"已保存输入PIL图像 {cam_name}_0 到: {pil_path}")
                
                images.append(pil_img)
            except Exception as e:
                print(f"转换 {cam_name} 图像为PIL格式时出错: {e}")
                images.append(PImage.new('RGB', (640, 480), color=(128, 128, 128)))
    
    # 添加第三个虚拟相机图像 (t-1)
    virtual_cam_img = PImage.new('RGB', (640, 480), color=(128, 128, 128))
    if save_images:
        virtual_path = os.path.join(DEBUG_DIR, f'pil_input_virtual_cam_0.jpg')
        virtual_cam_img.save(virtual_path)
        print(f"已保存虚拟相机图像到: {virtual_path}")
    images.append(virtual_cam_img)
    
    # 第二帧三个相机 (t)
    # 真实相机: camera_high 和 camera_right_wrist
    for cam_name in config['camera_names']:
        cv_img = observation_window[-1]['images'][cam_name]
        if cv_img is None:
            images.append(PImage.new('RGB', (640, 480), color=(128, 128, 128)))
        else:
            try:
                if not isinstance(cv_img, np.ndarray):
                    cv_img = np.array(cv_img)
                
                if cv_img.dtype != np.uint8:
                    cv_img = cv_img.astype(np.uint8)
                
                rgb_img = cv_img[:, :, ::-1]
                pil_img = PImage.fromarray(rgb_img)
                
                if save_images:
                    pil_path = os.path.join(DEBUG_DIR, f'pil_input_{cam_name}_1.jpg')
                    pil_img.save(pil_path)
                    print(f"已保存输入PIL图像 {cam_name}_1 到: {pil_path}")
                
                images.append(pil_img)
            except Exception as e:
                print(f"转换 {cam_name} 图像为PIL格式时出错: {e}")
                images.append(PImage.new('RGB', (640, 480), color=(128, 128, 128)))
    
    # 添加第三个虚拟相机图像 (t)
    virtual_cam_img = PImage.new('RGB', (640, 480), color=(128, 128, 128))
    if save_images:
        virtual_path = os.path.join(DEBUG_DIR, f'pil_input_virtual_cam_1.jpg')
        virtual_cam_img.save(virtual_path)
        print(f"已保存虚拟相机图像到: {virtual_path}")
    images.append(virtual_cam_img)
    
    # 获取关节状态
    proprio = observation_window[-1]['qpos']
    
    # 自动补充抓夹位置（第8个值为0）
    print(f"原始关节状态形状: {proprio.shape}")
    if proprio.shape[0] == 7:
        print("检测到7个关节位置，自动补充抓夹位置为0")
        proprio_with_gripper = torch.zeros(8, device=proprio.device, dtype=proprio.dtype)
        proprio_with_gripper[:7] = proprio
        proprio = proprio_with_gripper
        print(f"补充后的关节状态: {proprio}")
    
    # 扩展维度以匹配模型输入要求 [1, 8]
    proprio = proprio.unsqueeze(0)
    print(f"输入模型的关节状态形状: {proprio.shape}")
    print(f"图像列表长度: {len(images)}")
    
    print("开始模型推理...")
    # 生成动作序列
    try:
        actions = policy.step(
            proprio=proprio,
            images=images,
            text_embeds=lang_embeddings 
        ).squeeze(0).cpu().numpy()
        
        # 确保动作序列包含夹爪控制值（第8个值）
        if actions.shape[1] < 8:
            print(f"警告：模型输出动作只有 {actions.shape[1]} 个值，需要添加夹爪值")
            # 添加默认夹爪值（0表示打开）
            gripper_values = np.zeros((actions.shape[0], 1))
            actions = np.hstack((actions, gripper_values))
            print(f"已添加默认夹爪值，动作形状更新为: {actions.shape}")
        
        # 输出第一个动作用于调试
        print(f"第一个动作: {actions[0]}, 夹爪值: {actions[0][7]}")
        
        print(f"模型推理时间: {time.time() - time1} 秒")
        return actions
        
    except Exception as e:
        print(f"模型推理出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    
# 修改后的执行动作函数 - 添加夹爪控制并支持批量执行
def execute_actions(actions, franka_controller, ctrl_freq=25, batch_size=50):
    """执行模型生成的动作序列，包括夹爪控制
    
    Args:
        actions: 动作序列
        franka_controller: Franka控制器实例
        ctrl_freq: 控制频率
        batch_size: 一次执行的动作数量
    """
    try:
        print("开始执行动作序列...")
        
        # 计算要执行的总动作数
        total_actions = len(actions)
        print(f"动作序列总长度: {total_actions}")
        
        # 初始化夹爪状态 - 默认打开
        current_gripper_state = "close"
        franka_controller.set_gripper_positions(action="close")
        print("夹爪初始化为打开状态")
        
        # 分批执行动作
        remaining = total_actions
        start_idx = 0
        
        while remaining > 0:
            # 确定当前批次的动作数量
            current_batch_size = min(batch_size, remaining)
            end_idx = start_idx + current_batch_size
            
            print(f"\n执行动作批次: {start_idx+1} 到 {end_idx} (共 {total_actions})")
            
            # 执行当前批次的动作
            for i in range(start_idx, end_idx):
                # 获取当前动作 - 前7个值作为关节位置
                joint_positions = actions[i][:7]
                
                # 获取夹爪控制值 - 第8个值（如果存在）
                gripper_value = actions[i][7] if len(actions[i]) > 7 else 0.0
                
                print(f"执行动作 {i+1}/{total_actions}: {joint_positions}, 夹爪值: {gripper_value}")
                
                # 发送关节命令
                franka_controller.set_joint_positions(joint_positions.tolist())
                
                # 根据夹爪值控制夹爪
                desired_gripper_state = "close" if gripper_value > 0.5 else "open"
                
                # 只有当夹爪状态需要变化时才发送命令
                if desired_gripper_state != current_gripper_state:
                    print(f"夹爪状态变化: {current_gripper_state} -> {desired_gripper_state}")
                    franka_controller.set_gripper_positions(action=desired_gripper_state)
                    current_gripper_state = desired_gripper_state
                
                # 等待一个控制周期
                franka_controller.sleep()
            
            # 更新剩余动作数和起始索引
            remaining -= current_batch_size
            start_idx = end_idx
            
            # 如果还有剩余动作，询问是否继续
            if remaining > 0:
                next_batch_size = min(batch_size, remaining)
                continue_execution = input(f"已执行 {end_idx}/{total_actions} 个动作。是否继续执行下一批 {next_batch_size} 个动作? (y/n): ").lower().strip()
                if continue_execution != 'y':
                    print("用户请求停止执行")
                    break
        
        print("动作序列执行完成")
        return True
        
    except Exception as e:
        print(f"执行动作时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


# 修改主测试函数
def test_inference(args):
    global lang_embeddings
    global franka_controller
    
    # 初始化配置
    config = get_config(args)
    max_publish_step = config['episode_len']
    chunk_size = config['chunk_size']
    
    # 设置每批执行的步数
    batch_steps = 50
    
    # 初始化相机
    print("正在初始化RealSense相机...")
    pipelines = initialize_realsense()
    if pipelines is None or len(pipelines) < 2:
        print("无法初始化两个相机，测试退出")
        return
        
    # 初始化Franka控制器
    print("正在初始化Franka控制器...")
    franka_controller = initialize_franka_controller(rate=args.ctrl_freq)
    if franka_controller is None:
        print("无法初始化Franka控制器，测试退出")
        for pipeline in pipelines:
            pipeline.stop()
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
        input("系统已初始化。按Enter键开始推理测试...")
        
        # 获取初始关节位置作为前一个动作
        initial_joint_positions = get_franka_joint_positions(franka_controller)
        if initial_joint_positions is None:
            print("无法获取初始关节位置，使用零向量")
            pre_action = np.zeros(config['state_dim'])
        else:
            pre_action = initial_joint_positions
            
        print(f"初始关节位置: {pre_action}")
        
        # 更新观察窗口 - 获取两帧图像和关节数据
        print("获取第一帧相机图像和关节数据...")
        success = update_observation_window(config, pipelines, franka_controller)
        if not success:
            print("无法获取相机图像或关节数据，测试退出")
            return
            
        print("等待获取第二帧...")
        time.sleep(0.5)  # 等待获取第二帧
        
        success = update_observation_window(config, pipelines, franka_controller)
        if not success:
            print("无法获取第二帧相机图像或关节数据，测试退出")
            return
        
        # 初始化夹爪状态 - 默认打开
        current_gripper_state = "open"
        franka_controller.set_gripper_positions(action="open")
        print("夹爪已初始化为打开状态")
        
        # 主循环 - 添加夹爪控制
        print(f"开始执行最大 {max_publish_step} 步的推理循环...")
        t = 0
        action_buffer = np.zeros([chunk_size, config['state_dim']])
        
        with torch.inference_mode():
            while t < max_publish_step and not rospy.is_shutdown():
                # 确认执行下一批动作
                if t % batch_steps == 0 and t > 0:
                    if not args.execute_actions:  # 如果不是自动执行模式
                        continue_execution = input(f"已完成 {t} 步。执行下一批 {batch_steps} 步? (y/n): ").lower().strip()
                        if continue_execution != 'y':
                            print("用户请求停止执行")
                            break
                
                # 更新观察窗口 - 获取最新的相机图像和关节数据
                success = update_observation_window(config, pipelines, franka_controller)
                if not success:
                    print(f"第 {t} 步无法更新观察窗口，跳过此步")
                    time.sleep(0.1)
                    continue
                
                # 当到达action chunk的结尾时执行新的推理
                if t % chunk_size == 0:
                    print(f"步骤 {t}: 执行新的推理...")
                    actions = inference_fn(config, policy, t, save_images=(t % 10 == 0))
                    if actions is None:
                        print(f"步骤 {t}: 推理失败，跳过此步")
                        time.sleep(0.1)
                        continue
                    action_buffer = actions.copy()
                
                # 获取当前动作
                action = action_buffer[t % chunk_size]
                
                # 提取夹爪控制信号 - 第8个值
                gripper_value = action[7] if len(action) > 7 else 0.4
                print(f"步骤 {t}: 执行动作 {action[:7]}, 夹爪值: {gripper_value}")
                
                # 发送关节命令 - 只使用前7个值
                joint_positions = action[:7]
                franka_controller.set_joint_positions(joint_positions.tolist())
                
                # 根据夹爪值控制夹爪
                desired_gripper_state = "close" if gripper_value > 0.7 else "open"
              

                # 只有当夹爪状态需要变化时才发送命令
                if desired_gripper_state != current_gripper_state:
                    print(f"夹爪状态变化: {current_gripper_state} -> {desired_gripper_state}")
                    franka_controller.set_gripper_positions(action=desired_gripper_state)
                    current_gripper_state = desired_gripper_state
                
                # 更新前一个动作
                pre_action = action.copy()
                
                # 等待一个控制周期
                franka_controller.sleep()
                
                t += 1
                print(f"已完成推理步骤 {t}/{max_publish_step}")
            
        print(f"推理循环完成，共执行了 {t} 步")
            
    finally:
        # 清理资源
        for pipeline in pipelines:
            pipeline.stop()
        if franka_controller is not None:
            franka_controller.cleanup()
        print("测试完成，所有资源已释放")
        print(f"所有调试图像都保存在 {DEBUG_DIR} 目录中")


# 修改命令行参数解析函数，添加批处理步数参数
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
    
    parser.add_argument('--batch_steps', action='store', type=int,
                        help='Number of steps to execute in each batch',
                        default=50, required=False)
    
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", 
                        help='Path to the config file')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='Name or path to the pretrained model')
    
    parser.add_argument('--lang_embeddings_path', type=str, required=True, 
                        help='Path to the pre-encoded language instruction embeddings')
    
    parser.add_argument('--execute_actions', action='store_true', 
                        help='Automatically execute generated actions without asking')
    return parser.parse_args()
    

if __name__ == '__main__':
    # 解析命令行参数
    args = get_arguments()
    
    # 设置随机种子（如果提供）
    if args.seed is not None:
        set_seed(args.seed)
    
    # 执行主测试函数
    test_inference(args)