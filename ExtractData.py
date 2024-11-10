from flask import Flask, jsonify, request
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
from threading import Thread
from torch.utils.tensorboard import SummaryWriter

write = SummaryWriter("runs/TrainingMetrics")

app = Flask(__name__)

state_data_path = 'state_data.json'
reward_data_path = 'reward_data.json'

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim=2):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x
    
    def act(self, state):
        x = self.forward(state)
        action = torch.tanh(self.actor(x))
        throttle, steer = action[0].item(), action[1].item()
        throttle = 1 if throttle > 0.5 else -1 if throttle -0.5 else 0
        steer = 1 if steer > 0.6 else -1 if steer < -0.5 else 0
        return throttle, steer
    
    def evaluate(self, state):
        x = self.forward(state)
        return self.critic(x)

model = ActorCritic(input_dim=4)
optimizer = optim.Adam(model.parameters(), lr=0.001)
clip_epsilon = 0.2
gamma = 0.99
lambda_gae = 0.95

def load_data(path):
    try:
        with open(path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        return None

def clear_data(path):
    with open(path, 'w') as file:
        json.dump({}, file)

def process_reward(reward, state):
    speed = state['speed']
    steer = state["steer"]
    throttle = state["throttle"]
    collision = state.get('collision', False)
    lap_completion = state.get('lap_completion', False)

    total_reward = reward

    if 0.5 <= speed <= 0.8:
        total_reward += 1
    elif speed > 0.8:
        total_reward += 0.5
    
    if collision:
        total_reward -= 2

    if lap_completion:
        total_reward += 10

    return total_reward

def compute_gae(rewards, values, gamma, lambda_gae):
    advantages = []
    advantage = 0
    for i in reversed(range(len(rewards))):
        td_error = rewards[i] + gamma * values[i + 1] - values[i]
        advantage = td_error + gamma * lambda_gae * advantage
        advantages.insert(0, advantage)
    return advantages

def train_model():
    batch_states, batch_actions, batch_rewards, batch_values = [], [], [], []
    global_step = 0

    while True:
        state_data = load_data(state_data_path)
        reward_data = load_data(reward_data_path)

        if state_data:
            state = torch.tensor([state_data['distance'], state_data['speed'], state_data['steer'], state_data['throttle']], dtype=torch.float32)
            with torch.no_grad():
                throttle, steer = model.act(state)
            batch_states.append(state)
            batch_actions.append(torch.tensor([throttle, steer], dtype=torch.float32))
            batch_values.append(model.evaluate(state))
            clear_data(state_data_path)

        if reward_data:
            reward = reward_data.get('reward', 0)
            processed_reward = process_reward(reward, state_data)
            batch_rewards.append(processed_reward)
            clear_data(reward_data_path)

        if len(batch_states) >= 32:
            with torch.no_grad():
                next_value = model.evaluate(state)
            batch_values.append(next_value)

            advantages = compute_gae(batch_rewards, batch_values, gamma, lambda_gae)
            advantages = torch.tensor(advantages, dtype=torch.float32)
            batch_states = torch.stack(batch_states)
            batch_actions = torch.stack(batch_actions)
            batch_old_log_probs = torch.log(batch_actions)

            for _ in range(4):
                values = model.evaluate(batch_states)
                log_probs = torch.log(batch_actions)
                ratios = torch.exp(log_probs - batch_old_log_probs.detach())

                surrogate1 = ratios * advantages
                surrogate2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss = ((values.squeeze() - torch.tensor(batch_rewards, dtype=torch.float32)) ** 2).mean()
                loss = policy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                write.add_scalar("Loss/Policy", policy_loss.item(), global_step)
                write.add_scalar("Loss/Value", value_loss.item(), global_step)
                write.add_scalar("Loss/Total", loss.item(), global_step)
                write.add_scalar("Rewards/Average", sum(batch_rewards)/len(batch_rewards))

                global_step += 1

            batch_states, batch_actions, batch_rewards, batch_values = [], [], [], []

        time.sleep(0.1)


@app.route('/state', methods=["POST"])
def receive_state():
    data = request.get_json()
    with open(state_data_path, "w") as file:
        json.dump(data, file)

    state = torch.tensor([data['distance'], data['speed'], data['steer'], data['throttle']], dtype=torch.float32)
    throttle, steer = model.act(state)
    
    return jsonify({"status": "State received", "throttle": throttle, "steer": steer}), 200

@app.route('/reward', methods=["POST"])
def receive_reward():
    data = request.get_json()
    with open(reward_data_path, 'w') as file:
        json.dump(data, file)
    return jsonify({"status": "Reward received"}), 200

if __name__ == '__main__':
    training_thread = Thread(target=train_model)
    training_thread.start()
    app.run(debug=True)
