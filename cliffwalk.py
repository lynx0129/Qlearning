import time
import gymnasium as gym
import numpy as np
import pygame
import json
import argparse
import sys

def shutdown(env):
    """
    关闭环境和 pygame。

    :param env: Gym 环境对象
    """
    env.close()
    pygame.quit()
    sys.exit(0)



def render_rotated_mirrored_frame(env, screen, sleep_time):
    """
    获取环境图像并进行旋转+镜像处理，然后在 pygame 屏幕上显示。

    :param env: Gym 环境对象
    :param screen: Pygame 屏幕对象
    :param sleep_time: 每帧显示的时间间隔（秒）
    """
    frame = env.render()
    frame = np.array(frame)

    # 旋转90度（顺时针），然后进行镜像对调
    rotated_frame = np.rot90(frame, 1)
    mirrored_frame = np.flip(rotated_frame, axis=0)

    # 转换为 pygame Surface 并显示
    frame_surface = pygame.surfarray.make_surface(mirrored_frame)
    screen.blit(frame_surface, (0, 0))
    pygame.display.flip()
    time.sleep(sleep_time)

def train_q_learning(env, num_episodes=10, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
    """
    使用 Q-Learning 算法训练智能体。

    :param env: Gym 环境对象
    :param num_episodes: 训练轮数
    :param alpha: 学习率
    :param gamma: 折扣因子
    :param epsilon: 初始探索率
    :param epsilon_decay: 探索率衰减因子
    :param epsilon_min: 最小探索率
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))  # 初始化 Q 表
    goal_state = n_states - 1  # 终点状态

    print("开始训练……")
    for episode in range(num_episodes):
        reset_return = env.reset()
        if isinstance(reset_return, tuple):
            state = reset_return[0]
        else:
            state = reset_return
        done = False

        while not done:
            # ε-贪心策略选择动作
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            # 执行动作，获取反馈
            step_return = env.step(action)
            if len(step_return) == 4:
                next_state, reward, done, info = step_return
            else:
                next_state, reward, done, truncated, info = step_return

            # 自定义 Cliff Walking 的奖励策略（原始奖励已经适合 Q-Learning）
            if done and next_state != goal_state:  # 掉落悬崖
                reward = -100

            # 更新 Q 表
            best_next_action = np.argmax(Q[next_state, :])
            Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])
            state = next_state

        
        if args.print_q:
            print(Q)
        
        # 每个 episode 结束后降低探索率
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode + 1}/{num_episodes} 完成.")

    print("🎉 训练完成！")
    return Q

parser = argparse.ArgumentParser()
parser.add_argument("--num_episodes", "-np", type=int, default=100, help="Number of training episodes")
parser.add_argument("--print_q", "-pq", type=bool, default=False, help="Whether to print Q-table")
args = parser.parse_args()

# ----------------------------
# 初始化环境和 pygame
# ----------------------------

env = gym.make("CliffWalking-v1", render_mode="rgb_array")

# 获取状态空间大小
n_states = env.observation_space.n
n_actions = env.action_space.n
goal_state = n_states - 1  # 终点状态

# 获取环境的初始图像
init_state = env.reset()
frame = env.render()
height, width, _ = frame.shape


# ----------------------------
# Q-Learning 参数设置
# ----------------------------
Q = np.zeros((n_states, n_actions))  # 初始化 Q 表

# 从 JSON 文件加载 Q-Learning 参数
with open("qlearning_config.json", "r") as f:
    config = json.load(f)

alpha = config["alpha"]
gamma = config["gamma"]
epsilon = config["epsilon"]
epsilon_decay = config["epsilon_decay"]
epsilon_min = config["epsilon_min"]

num_episodes = args.num_episodes  # 训练轮数


# ----------------------------
# 训练 Q-Learning 并使用 pygame 可视化训练过程
# ----------------------------
Q = train_q_learning(env, num_episodes, alpha, gamma, epsilon, epsilon_decay, epsilon_min)

# 初始化 pygame
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("CliffWalking Q-Learning Training with pygame")


# ----------------------------
# 利用 pygame 可视化训练后智能体的最优路径
# ----------------------------
print("展示训练后智能体执行最优策略……")
reset_return = env.reset()
if isinstance(reset_return, tuple):
    state = reset_return[0]
else:
    state = reset_return
done = False

render_rotated_mirrored_frame(env, screen, 0.5)

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            shutdown(env)



    # 选择 Q 表中的最佳动作
    action = np.argmax(Q[state, :])
    step_return = env.step(action)
    if len(step_return) == 4:
        state, reward, done, info = step_return
    else:
        state, reward, done, truncated, info = step_return
        done = done or truncated

    render_rotated_mirrored_frame(env, screen, 0.5)

# 保持窗口打开，等待用户关闭
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
