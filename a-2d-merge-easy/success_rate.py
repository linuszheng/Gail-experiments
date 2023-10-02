from stable_baselines3.ppo import MlpPolicy
import torch


device = torch.device('cpu')
model = MlpPolicy.load("best_model", device=device)
print(model.predict([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])[0])




