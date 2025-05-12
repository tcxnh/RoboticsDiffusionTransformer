#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
from typing import List

## 示例点
START_POSE = [0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397]
EXAMPLE_POSE = [0.26975516089480217,-0.3673510828805246,0.06568009957317089,-2.483767545555859,-0.0008299042695231822,2.237880261282787,0.8335631263531316]

TEST_POSE = [-0.20949405,0.30368334,-0.01772219,-2.1737068,0.14235234 ,2.5003448, -0.43730223] 
TSET_POSE2= [0.12731111,0.32844487, -0.0932996,-2.1159036,0.04111218,2.4346464,0.76795506]
TEST_POSE3 = [-0.03813702, 0.21082762,0.0068405,-2.1247964,0.01470166,2.3489528,-0.135988]
TEST_POSE4 = [-0.10904336,0.4151102, -0.05362144,-2.2315097,0.04991567,2.6517367,-0.33372545]
## GRIPPER
GRIPPER_OPEN = [0.0, 0.0]
GRIPPER_CLOSE = [1.0, 0.0]

class FrankaIOController:
    def __init__(self, rate = 10):
        rospy.init_node('franka_controller', anonymous=True)

        self.rate = rospy.Rate(rate)

        # 关节角度发布器
        self.joint_pub = rospy.Publisher('/io_teleop/joint_cmd', JointState, queue_size=10)
        
        # 夹爪控制发布器（假设夹爪是单自由度）
        self.gripper_pub = rospy.Publisher('/io_teleop/target_gripper_status', JointState, queue_size=10)

        # 添加关节状态订阅器
        self.joint_states_sub = rospy.Subscriber("/joint_states", JointState, self.joint_states_callback)
        
        # 用于存储当前关节状态的变量
        self.current_joint_positions = None
        self.is_joint_state_received = False

        # 等待注册完成
        rospy.sleep(0.5)

        self.init_gripper()
        self.init_joint()

    def joint_states_callback(self, msg):
        """
        回调函数，用于接收关节状态信息
        """
        # 保存关节位置
        self.current_joint_positions = list(msg.position)
        self.is_joint_state_received = True
        # 可以根据需要增加对关节速度和力矩的保存
        # self.current_joint_velocities = list(msg.velocity)
        # self.current_joint_efforts = list(msg.effort)

    def get_current_joint_positions(self):
        """
        获取当前关节位置的函数
        
        Returns:
            List: 当前关节位置列表，如果未接收到数据则返回None
        """
        if not self.is_joint_state_received:
            rospy.logwarn("No joint state has been received yet.")
            return None
        return self.current_joint_positions

    def wait_for_joint_state(self, timeout=5.0):
        """
        等待接收到关节状态数据
        
        Args:
            timeout (float): 超时时间（秒）
            
        Returns:
            bool: 是否成功接收到关节状态
        """
        start_time = rospy.Time.now()
        while not self.is_joint_state_received:
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logwarn(f"Timed out waiting for joint state after {timeout} seconds.")
                return False
            rospy.sleep(0.1)
        return True

    def init_joint(self):
        # 初始化关节状态消息
        self.joint_cmd = JointState()
        self.joint_cmd.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self.joint_cmd.position = START_POSE  # 初始位置归零
        self.joint_cmd.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 可选：速度控制
        self.joint_cmd.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 可选：力矩控制

    def init_gripper(self):
        self.gripper_cmd = JointState()
        self.gripper_cmd.name = ['joint1', 'joint2']
        self.gripper_cmd.position = GRIPPER_OPEN  # 默认夹爪打开
        self.gripper_cmd.velocity = [0.0, 0.0]  # 修正为与name长度匹配
        self.gripper_cmd.effort = [0.0, 0.0]    # 修正为与name长度匹配
    
    def set_joint_positions(self, positions: List):
        """关节透传函数
        """
        if len(positions) != 7:
            rospy.logerr("Invalid joint positions! Expected 7 values.")
            return
        
        self.joint_cmd.position = positions
        self.joint_cmd.header.stamp = rospy.Time.now()
      
        self.joint_pub.publish(self.joint_cmd)
        rospy.loginfo(f"Joints set to: {positions}")

    def set_gripper_positions(self, action: str = "open"):
        """设置夹爪状态
        - param: close | open
        """
        if action not in ["open", "close"]:
            rospy.logerr("Invalid gripper positions! Expected \'open\' or \'close\'")
            return
        
        # 设置关节位置（单位：弧度）
        if action == "open":
            self.gripper_cmd.position = GRIPPER_OPEN  # 示例值
        elif action == "close":
            self.gripper_cmd.position = GRIPPER_CLOSE  # 示例值
        else:
            rospy.logerr("Invalid gripper positions! Expected \'open\' or \'close\'")
            return
        
        self.gripper_cmd.header.stamp = rospy.Time.now()
        self.gripper_pub.publish(self.gripper_cmd)
        rospy.loginfo(f"Gripper set to: {action}")
    
    def sleep(self):
        """设置发送频率时需调用，和 set_joint_positions 配合使用
        """
        self.rate.sleep()
    
    def cleanup(self):
        """清理资源，关闭发布器"""
        self.joint_pub.unregister()
        self.gripper_pub.unregister()
        self.joint_states_sub.unregister()  # 取消订阅
        rospy.loginfo("FrankaIOController shutdown.")

# if __name__ == '__main__':
#     try:
#         robot = FrankaIOController(rate=25)
        
#         # 关节透传 谨慎操作！！！
#         # 示例中两个点可以使用，但推荐先使用 move_to_start.py 到达开始位置
#         robot.set_joint_positions(TEST_POSE4)
#         robot.sleep()
        
#         # 等待接收关节状态
#         if robot.wait_for_joint_state(timeout=3.0):
#             # 打印当前关节位置
#             current_joints = robot.get_current_joint_positions()
#             rospy.loginfo(f"Current joint positions: {current_joints}")

#         # robot.set_gripper_positions(action="open")
#         # rospy.sleep(2)  # 等待动作执行
        
#         # rospy.spin()  # 保持节点运行（如果需要订阅回调）
        
#     except rospy.ROSInterruptException:
#         pass
#     finally:
#         robot.cleanup()