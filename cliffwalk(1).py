
import time

import gymnasium as gym
import numpy as np
import pygame

WINDOW_SCALE = 2
FRAME_INTERVAL = 0.5
IDLE_FPS = 30
SEED = 0


def render_frame(env, screen, sleep_time):
    frame = np.asarray(env.render())
    surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
    pygame.transform.scale(surface, screen.get_size(), screen)
    pygame.display.flip()
    time.sleep(sleep_time)


def quit_all(env):
    env.close()
    pygame.quit()
    raise SystemExit


env = gym.make("CliffWalking-v1", render_mode="rgb_array")

n_states = env.observation_space.n
n_actions = env.action_space.n

np.random.seed(SEED)
env.reset(seed=SEED)
env.action_space.seed(SEED)

height, width, _ = np.asarray(env.render()).shape

pygame.init()
screen = pygame.display.set_mode((width * WINDOW_SCALE, height * WINDOW_SCALE))
pygame.display.set_caption("CliffWalking Q-Learning Training with pygame")

Q = np.zeros((n_states, n_actions))
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 1000

print("开始训练……")
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_all(env)

        if np.random.rand() < epsilon:
            action = int(env.action_space.sample())
        else:
            action = int(np.argmax(Q[state, :]))

        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated

        best_next_action = int(np.argmax(Q[next_state, :]))
        Q[state, action] += alpha * (
            reward + gamma * Q[next_state, best_next_action] - Q[state, action]
        )
        state = next_state

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes} 完成，ε = {epsilon:.4f}")

print("🎉 训练完成！")

print("展示训练后智能体执行最优策略……")
state, _ = env.reset()
done = False

render_frame(env, screen, FRAME_INTERVAL)

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_all(env)

    action = int(np.argmax(Q[state, :]))
    state, reward, done, truncated, _ = env.step(action)
    done = done or truncated

    render_frame(env, screen, FRAME_INTERVAL)

clock = pygame.time.Clock()
running = True
while running:
    clock.tick(IDLE_FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

quit_all(env)
