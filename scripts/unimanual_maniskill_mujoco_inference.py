#!/usr/bin/env python
# -- coding: UTF-8
"""
单臂推理测试代码 - 使用MuJoCo环境替代真实机器人
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

# 导入MuJoCo相关库（替代PyRealSense2和FrankaIOController）
import mujoco
from mujoco import viewer
from scripts.maniskill_model_no_lan_cal import create_model, RoboticDiffusionTransformerModel

# 创建保存图像的目录
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

# 相机名称保持不变
CAMERA_NAMES = ['cam_right_wrist', 'cam_high']

observation_window = None
lang_embeddings = None
mujoco_sim = None  # MuJoCo模拟器实例，替代franka_controller


# Initialize the model (保持不变)
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
        'state_dim': 8,  # 保持为8（7个关节+1个夹爪）
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,  
    }
    return config


def save_pil_image(filepath, img_data):
    """使用PIL保存图像"""
    try:
        if img_data is None:
            print(f"警告: 尝试保存空图像到 {filepath}")
            return False
            
        # 确保数据类型是uint8
        if isinstance(img_data, np.ndarray) and img_data.dtype != np.uint8:
            img_data = img_data.astype(np.uint8)
        
        # 转换为PIL图像并保存
        pil_img = PImage.fromarray(img_data)
        pil_img.save(filepath)
        print(f"已保存图像到: {filepath}")
        return True
            
    except Exception as e:
        print(f"保存图像过程中出错: {e}")
        return False


def initialize_mujoco_sim(xml_path):
    """初始化MuJoCo环境并设置相机"""
    try:
        print("正在初始化MuJoCo环境...")
        
        # 设置OpenGL渲染后端为EGL，适用于有GPU的系统
        os.environ['MUJOCO_GL'] = 'egl'  # 使用EGL进行GPU加速渲染
        
        # 加载模型
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        
        # 创建交互式视图
        viewer_obj = None
        try:
            viewer_obj = mujoco.viewer.launch_passive(model, data)
            print("交互式视图创建成功")
        except Exception as e:
            print(f"创建交互式视图时出错: {e}")
            print("将继续运行但没有交互式视图")
        
        # 重置模拟器
        mujoco.mj_resetData(model, data)
        
        # 获取模型中的相机列表
        cam_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) 
                    for i in range(model.ncam)]
        print(f"模型中定义的相机: {cam_names}")
        
        # 检查我们需要的相机是否存在
        required_cams = ['cam_high', 'cam_right_wrist']
        missing_cams = [cam for cam in required_cams if cam not in cam_names]
        if missing_cams:
            print(f"警告: 模型中缺少以下相机: {missing_cams}")
            print("请确保在XML文件中正确定义了所有必需的相机")
        
        # 创建渲染器，使用原始分辨率，因为有GPU支持
        renderers = {}
        camera_ids = {}
        
        for cam_name in required_cams:
            try:
                if cam_name not in cam_names:
                    print(f"跳过未定义的相机: {cam_name}")
                    renderers[cam_name] = None
                    camera_ids[cam_name] = -1
                    continue
                
                # 获取相机ID
                cam_id = cam_names.index(cam_name)
                camera_ids[cam_name] = cam_id
                
                # 创建渲染器，采用完整分辨率
                renderer = mujoco.Renderer(model, width=640, height=480)
                
                # 测试渲染
                try:
                    renderer.update_scene(data, camera=cam_id)
                    test_img = renderer.render()
                    
                    if test_img is None or test_img.size == 0:
                        raise ValueError(f"相机 {cam_name} 渲染返回空图像")
                    
                    # 保存测试图像用于调试
                    test_img_path = os.path.join(DEBUG_DIR, f"test_{cam_name}.jpg")
                    cv2.imwrite(test_img_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
                    print(f"成功创建 {cam_name} 的渲染器，测试图像已保存到 {test_img_path}")
                    
                    renderers[cam_name] = renderer
                except Exception as e:
                    print(f"{cam_name} 渲染器测试失败: {e}")
                    import traceback
                    traceback.print_exc()
                    renderers[cam_name] = None
            except Exception as e:
                print(f"创建 {cam_name} 渲染器时出错: {e}")
                renderers[cam_name] = None
        
        # 如果所有渲染器都初始化失败，尝试使用不同的渲染后端
        if all(renderer is None for renderer in renderers.values()):
            print("警告: GPU渲染初始化失败，尝试使用软件渲染...")
            os.environ['MUJOCO_GL'] = 'osmesa'
            
            for cam_name in required_cams:
                if cam_name not in cam_names:
                    continue
                    
                try:
                    cam_id = cam_names.index(cam_name)
                    renderer = mujoco.Renderer(model, width=640, height=480)
                    renderer.update_scene(data, camera=cam_id)
                    test_img = renderer.render()
                    
                    if test_img is not None and test_img.size > 0:
                        test_img_path = os.path.join(DEBUG_DIR, f"test_osmesa_{cam_name}.jpg")
                        cv2.imwrite(test_img_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
                        print(f"使用软件渲染成功创建 {cam_name} 的渲染器")
                        renderers[cam_name] = renderer
                except Exception as e:
                    print(f"软件渲染也失败: {e}")
        
        # 执行一些初始步骤来稳定模拟器
        for _ in range(10):
            mujoco.mj_step(model, data)
        
        return {
            'model': model,
            'data': data,
            'viewer': viewer_obj,
            'renderers': renderers,
            'camera_ids': camera_ids,
            'resolution': (640, 480)  # 存储渲染分辨率
        }
        
    except Exception as e:
        print(f"初始化MuJoCo环境时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_mujoco_camera_images(mujoco_sim):
    """从MuJoCo获取相机图像"""
    images = []
    
    for cam_name in ['cam_high', 'cam_right_wrist']:
        try:
            # 检查渲染器是否可用
            if (cam_name not in mujoco_sim['renderers'] or 
                mujoco_sim['renderers'][cam_name] is None):
                print(f"警告: {cam_name} 渲染器不可用，使用占位图像")
                placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 128
                images.append(placeholder)
                continue
                
            renderer = mujoco_sim['renderers'][cam_name]
            cam_id = mujoco_sim['camera_ids'][cam_name]
            
            # 更新场景
            renderer.update_scene(mujoco_sim['data'], camera=cam_id)
            
            # 渲染图像
            img = renderer.render()
            
            # 保存调试图像
            timestamp = datetime.now().strftime("%H%M%S%f")
            img_path = os.path.join(DEBUG_DIR, f"mujoco_{cam_name}_{timestamp}.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            images.append(img)
                
        except Exception as e:
            print(f"获取 {cam_name} 图像时出错: {e}")
            placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 128
            images.append(placeholder)
    
    return images


# 获取MuJoCo关节位置 (替代get_franka_joint_positions)
def get_mujoco_joint_positions(mujoco_sim):
    """从MuJoCo环境获取当前关节位置"""
    try:
        # 获取关节位置 - 这里需要根据您的MuJoCo模型调整
        # 假设前7个值是机器人关节
        joint_positions = mujoco_sim['data'].qpos[:7].copy()
        
        # 转换为numpy数组
        joint_positions = np.array(joint_positions, dtype=np.float32)
        
        print(f"获取的MuJoCo关节位置: {joint_positions}")
        return joint_positions
        
    except Exception as e:
        print(f"获取MuJoCo关节位置时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


# 更新观察窗口 - 使用MuJoCo数据
def update_observation_window(config, mujoco_sim):
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
    
    # 获取MuJoCo相机图像
    print("正在获取MuJoCo相机图像...")
    images = get_mujoco_camera_images(mujoco_sim)
    if len(images) < 2:
        print(f"错误: 只获取到 {len(images)} 个相机图像，需要2个")
        return False
    
    # 处理图像
    processed_images = {}
    for i, cam_name in enumerate(config["camera_names"]):
        print(f"正在处理相机 {cam_name} 的图像...")
        img = images[i]
        processed_images[cam_name] = img
    
    # 获取MuJoCo关节位置
    print("正在获取MuJoCo关节位置...")
    joint_positions = get_mujoco_joint_positions(mujoco_sim)
    
    if joint_positions is None:
        print("错误: 无法获取MuJoCo关节位置，使用零向量代替")
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


def execute_mujoco_actions(actions, mujoco_sim, ctrl_freq=25, batch_size=50):
    """在MuJoCo环境中执行模型生成的动作序列
    
    Args:
        actions: 动作序列
        mujoco_sim: MuJoCo模拟器实例
        ctrl_freq: 控制频率
        batch_size: 一次执行的动作数量
    """
    try:
        print("开始在MuJoCo环境中执行动作序列...")
        
        # 计算要执行的总动作数
        total_actions = len(actions)
        print(f"动作序列总长度: {total_actions}")
        
        # 初始化夹爪状态 - 默认打开
        current_gripper_state = 0.0  # 0表示打开，1表示关闭
        
        # 分批执行动作，但不再询问用户是否继续
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
                
                # 减少日志输出频率
                if i % 10 == 0:
                    print(f"执行动作 {i+1}/{total_actions}: {joint_positions}, 夹爪值: {gripper_value}")
                
                # 设置MuJoCo的关节位置目标
                mujoco_sim['data'].ctrl[:7] = joint_positions
                
                # 设置夹爪控制值
                gripper_target = 1.0 if gripper_value > 0.8 else 0.0
                mujoco_sim['data'].ctrl[7] = gripper_target
                
                # 前进仿真一步
                mujoco.mj_step(mujoco_sim['model'], mujoco_sim['data'])
                
                # 更新视图
                if mujoco_sim['viewer'] is not None:
                    mujoco_sim['viewer'].sync()
                
                # 控制仿真速度
                time.sleep(1.0 / ctrl_freq)
            
            # 更新剩余动作数和起始索引
            remaining -= current_batch_size
            start_idx = end_idx
            
            # 删除询问用户是否继续的代码，直接继续执行
        
        print("动作序列执行完成")
        return True
        
    except Exception as e:
        print(f"执行MuJoCo动作时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


# 修改推理函数 - 部分保持相同，只替换相关的MuJoCo部分
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


# 修改主测试函数 - 使用MuJoCo环境，去除每50步的确认
def test_inference(args):
    global lang_embeddings
    global mujoco_sim
    
    # 在代码中直接定义MuJoCo模型路径
    xml_path = "/home/ubuntu/桌面/franka_sim/franka_panda.xml"  
    
    # 初始化配置
    config = get_config(args)
    max_publish_step = config['episode_len']
    chunk_size = config['chunk_size']
    
    # 初始化MuJoCo环境
    print("正在初始化MuJoCo环境...")
    mujoco_sim = initialize_mujoco_sim(xml_path)
    if mujoco_sim is None:
        print("无法初始化MuJoCo环境，测试退出")
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
        initial_joint_positions = get_mujoco_joint_positions(mujoco_sim)
        if initial_joint_positions is None:
            print("无法获取初始关节位置，使用零向量")
            pre_action = np.zeros(config['state_dim'])
        else:
            pre_action = initial_joint_positions
            
        print(f"初始关节位置: {pre_action}")
        
        # 更新观察窗口 - 获取两帧图像和关节数据
        print("获取第一帧MuJoCo图像和关节数据...")
        success = update_observation_window(config, mujoco_sim)
        if not success:
            print("无法获取MuJoCo图像或关节数据，测试退出")
            return
            
        print("等待获取第二帧...")
        # 前进MuJoCo仿真一步
        mujoco.mj_step(mujoco_sim['model'], mujoco_sim['data'])
        if mujoco_sim['viewer'] is not None:
            mujoco_sim['viewer'].sync()
        
        success = update_observation_window(config, mujoco_sim)
        if not success:
            print("无法获取第二帧MuJoCo图像或关节数据，测试退出")
            return
        
        # 主循环 - 移除了每批次的用户确认
        print(f"开始执行最大 {max_publish_step} 步的推理循环...")
        t = 0
        action_buffer = np.zeros([chunk_size, config['state_dim']])
        
        with torch.inference_mode():
            while t < max_publish_step:
                # 移除批次确认代码，改为简单的进度日志
                if t % 50 == 0 and t > 0:
                    print(f"持续执行中...已完成 {t}/{max_publish_step} 步")
                
                # 更新观察窗口 - 获取最新的MuJoCo图像和关节数据
                success = update_observation_window(config, mujoco_sim)
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
                gripper_value = action[7] if len(action) > 7 else 0.0
                print(f"步骤 {t}: 执行动作 {action[:7]}, 夹爪值: {gripper_value}")
                
                # 设置MuJoCo的控制信号
                # 前7个值为关节位置
                mujoco_sim['data'].ctrl[:7] = action[:7]
                
                # 设置夹爪控制值
                gripper_target = 1.0 if gripper_value > 0.5 else 0.0
                mujoco_sim['data'].ctrl[7] = gripper_target
                
                # 前进MuJoCo仿真一步
                mujoco.mj_step(mujoco_sim['model'], mujoco_sim['data'])
                
                # 更新视图
                if mujoco_sim['viewer'] is not None:
                    mujoco_sim['viewer'].sync()
                
                # 等待一个控制周期
                time.sleep(1.0 / args.ctrl_freq)
                
                t += 1
                
                # 每20步输出一次简洁的进度信息，减少日志输出频率
                if t % 20 == 0:
                    print(f"已完成推理步骤 {t}/{max_publish_step}")
            
        print(f"推理循环完成，共执行了 {t} 步")
       
    finally:
        # 清理MuJoCo资源
        if mujoco_sim is not None:
            # 关闭交互式viewer
            if mujoco_sim['viewer'] is not None:
                try:
                    mujoco_sim['viewer'].close()
                    print("已关闭交互式viewer")
                except Exception as e:
                    print(f"关闭viewer时出错: {e}")
            
            # 清理渲染器 - 添加安全检查
            if 'renderers' in mujoco_sim:
                for cam_name, renderer in mujoco_sim['renderers'].items():
                    if renderer is not None:
                        try:
                            # 检查是否有我们添加的标志
                            if hasattr(renderer, '_gl_context_initialized') and renderer._gl_context_initialized:
                                # 安全地关闭渲染器
                                # 先移除可能导致问题的属性
                                if hasattr(renderer, '_gl_context'):
                                    renderer._gl_context = None
                                renderer.close()
                                print(f"已关闭相机 {cam_name} 的渲染器")
                            else:
                                print(f"跳过相机 {cam_name} 的渲染器关闭 (未初始化)")
                        except Exception as e:
                            print(f"关闭渲染器时出错: {e}")
        
        print("测试完成，所有资源已释放")
        print(f"所有调试图像都保存在 {DEBUG_DIR} 目录中")


# 命令行参数解析函数 - 不包含MuJoCo模型路径
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
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, 
                        help='Name or path to the pretrained model')
    
    parser.add_argument('--lang_embeddings_path', type=str, required=True, 
                        help='Path to the pre-encoded language instruction embeddings')
    
    return parser.parse_args()
    

if __name__ == '__main__':
    # 解析命令行参数
    args = get_arguments()
    
    # 设置随机种子（如果提供）
    if args.seed is not None:
        set_seed(args.seed)
    
    # 执行主测试函数
    test_inference(args)