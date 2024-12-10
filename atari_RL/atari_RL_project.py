# Install the required libraries:

# pip install threading numpy pygame
# pip install gymnasium[atari,accept-rom-license] stable-baselines3[extra] torch torchvision tensorboard sb3-contrib
# pip install opencv-python ale-py
# pip install optuna optuna-integration[pytorch_lightning]

# Run this command in the python terminal to start TensorBoard during RL training:
# tensorboard --logdir ./logs --port 6006

# Run this command in the python terminal to start TensorBoard during RL parameter optimization:
# tensorboard --logdir ./neptuna_logs --port 6007

# Adjustments:
# Modify render_freq to control how often gameplay is rendered.
# Increase time_steps for more extensive training.

# To run jupyter notebooks in Pycharm:
#
#  1. Install juypter (pip install jupyter or use the interface to install this python package)
#  2. In the console/terminal type "jupyter-notebook"
#  3. Access your .pynb file and folders using your computers web browser which will now be running the jupyter
#  4. Typical address for the web browser is http://localhost:8888/notebooks
#
import gymnasium as gym
import ale_py
from stable_baselines3.common.atari_wrappers import AtariWrapper, EpisodicLifeEnv
from sb3_contrib import RecurrentPPO, TRPO, QRDQN
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import optuna
import os
import pygame
import cv2
import json
import numpy as np
import queue
import re
import threading
import time

# Shared flag for stopping the thread
stop_event = threading.Event()


def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)


def select_model_train():
    algorithm_type_choice = input("Choose Algorithm:\n"
                                  "1. REINFORCE (Not Implemented)\n"
                                  "2. A2C\n"
                                  "3. DQN\n"
                                  "4. PPO\n"
                                  "5. TRPO\n"
                                  "6. QRDQN\n"
                                  "7. RecurrentPPO\n"
                                  "Enter your choice (Default 4): ")
    if not algorithm_type_choice:
        algorithm_type_choice = "4"
    algorithm_type_array = ["REINFORCE", "A2C", "DQN", "PPO", "TRPO", "QRDQN", "RPPO"]
    algorithm_choice = algorithm_type_array[int(algorithm_type_choice) - 1]
    if algorithm_choice == 1:
        print("REINFORCE not implemented. Choosing PPO.")
        algorithm_choice = "PPO"
    return algorithm_choice


def list_and_select_model(folder_path, file_extension=".zip"):
    """
    Lists all model files in a folder with a specified extension and allows the user to select one.

    Args:
        folder_path (str): Path to the folder containing model files.
        file_extension (str): Extension of the model files to list (default: ".zip").

    Returns:
        str: The full path to the selected model file.
    """
    # Get a list of files in the folder with the specified extension
    model_files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(file_extension)
    ]
    # Check if there are any model files
    if not model_files:
        print(f"No model files with extension '{file_extension}' found in {folder_path}.")
        return None
    model_files = sorted(model_files)
    # Display the list of model files
    print("Available Models:")
    for idx, model_file in enumerate(model_files, start=1):
        print(f"{idx}. {model_file}")
    # Prompt the user to select a file
    while True:
        try:
            choice = int(input(f"Select a model (1-{len(model_files)}): "))
            if 1 <= choice <= len(model_files):
                selected_model = os.path.join(folder_path, model_files[choice - 1])
                print(f"Selected Model: {selected_model}")
                return selected_model
            else:
                print(f"Please enter a number between 1 and {len(model_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


# RenderCallback: Renders the agent playing the game every 100,000 steps.
class RenderCallback(BaseCallback):
    def __init__(self, env, render_freq=100_000, **kwargs):
        super(RenderCallback, self).__init__(**kwargs)
        self.env = env
        self.render_freq = render_freq
        self.render_threshold = self.render_freq
        self.epoch_count = 0
        self.frame_wait_time_ms = 1
        self.frame_width = 600
        self.frame_height = 600
        self.max_timesteps = 100  # make smaller to reduce time observing agent play
        # Initialize pygame
        pygame.init()
        # Create a small pygame window
        self.screen = pygame.display.set_mode((self.frame_width, self.frame_height))

    def _on_step(self) -> bool:
        if self.num_timesteps > self.render_threshold:
            self.render_threshold += self.render_freq
            self.epoch_count += 1
            pygame.display.set_caption(f"RL Training Agent Play at Epoch {self.epoch_count}: {atari_game_id}")
            print(f"\nEpoch {self.epoch_count}: Rendering agent playing the game...")
            obs = self.env.reset()
            clock = pygame.time.Clock()  # Initialize the clock for frame rate control
            for _ in range(self.max_timesteps):  # Render a few steps of gameplay
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                action, _ = self.model.predict(obs)
                obs, _, done, _ = self.env.step(action)
                obs_image = self.env.render()  # Returns the rendered frame as an array
                # Rendering and resizing a frame
                obs_image = self.env.render()  # Returns the rendered frame as an array
                resized_frame = resize_frame(obs_image, width=self.frame_width, height=self.frame_height)
                # Convert obs to pygame surface and blit to screen
                # Render the game screen for the human player to see
                self.screen.blit(pygame.surfarray.make_surface(resized_frame.transpose((1, 0, 2))), (0, 0))
                # Update the display
                pygame.display.flip()
                # Limit frame rate to specific FPS
                clock.tick(5)
                if np.all(done):
                    break
        return True


# Callback to save the model periodically
class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, save_path, save_prefix="dl_atari_epoch_", verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_threshold = save_freq
        self.save_path = save_path
        self.epoch_count = 0
        self.save_prefix = save_prefix

    def _on_step(self) -> bool:
        # Check if the save interval has been reached
        if self.num_timesteps > self.save_threshold:
            self.save_threshold += save_freq
            self.epoch_count += 1
            save_file = os.path.join(self.save_path, f"{self.save_prefix}_atari_epoch_{self.epoch_count}.zip")
            # save_file = os.path.join(self.save_path, f"{self.model.__class__}_atari_epoch_{self.epoch_count}.zip")
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"Model saved at {save_file}")
        return True


# Custom callback for PyTorch-based TensorBoard logging
class TensorBoardModelCallback(BaseCallback):
    def __init__(self, model, frame_skip=1, log_interval=10, **kwargs):
        super(TensorBoardModelCallback, self).__init__(**kwargs)
        self.model = model
        self.log_interval = log_interval
        self.frame_skip = frame_skip
        self.episode_rewards = []
        self.episode_frame_lengths = []

    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info.keys():
                    reward = info["episode"]["r"]
                    frame_length = info["episode"]["l"] * self.frame_skip
                    self.episode_rewards.append(reward)
                    self.episode_frame_lengths.append(frame_length)
                    # self.writer.add_scalar("Episode Reward", reward, self.num_timesteps)
                    # self.writer.add_scalar("Episode Frame Length", length, self.num_timesteps)

        # Access the TensorBoard writer
        for logger_output in self.model.logger.output_formats:
            if hasattr(logger_output, 'writer'):
                writer = logger_output.writer  # This is the TensorBoard writer
                break
        else:
            writer = None

        # Log average reward to TensorBoard at intervals
        if len(self.episode_rewards) >= self.log_interval:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            avg_frame_length = sum(self.episode_frame_lengths) / len(self.episode_frame_lengths)
            self.episode_rewards.clear()  # Reset for the next interval
            self.episode_frame_lengths.clear()  # Reset for the next interval
            if self.verbose:
                print(f"Step: {self.num_timesteps}, Avg Episode Reward: {avg_reward:.2f}")
                print(f"Step: {self.num_timesteps}, Avg Episode Frame Length: {avg_frame_length:.2f}")

            if writer:
                # Log a custom metric (example: custom reward metric)
                writer.add_scalar("custom/mean_reward", avg_reward, self.num_timesteps)
                writer.add_scalar("custom/mean_episode_frame_length", avg_frame_length, self.num_timesteps)

        return True


# Function to create a monitored environment
def make_monitored_env(rank, log_dir=None, frame_skip=4, screen_size=84):
    """
    Utility function to create a monitored environment.

    Args:
        rank (int): Unique identifier for the environment instance.
        log_dir (str): Directory to save logs.

    Returns:
        function: A callable function that returns the wrapped environment.
    """

    def _init():
        environment_config = {
            "id": atari_game_id,  # Replace 'Breakout-v5' with another game if desired
            "render_mode": "rgb_array",  # None, "human", "rgb_array"
            "full_action_space": False,
            "repeat_action_probability": 0,
            "mode": 0
        }
        env = gym.make(**environment_config)
        # Each environment has its own log file
        # Preprocess Atari environment (resize, grayscale, etc.)
        if log_dir is not None:
            env = Monitor(env, os.path.join(log_dir, f"env_{rank}.log"))
        env = AtariWrapper(env, frame_skip=frame_skip, screen_size=screen_size)
        # Wrap with EpisodicLifeEnv to treat one life as an episode
        env = EpisodicLifeEnv(env)
        return env

    return _init


def make_env(frame_skip=4):
    # create environment
    environment_config = {
        # "id": "Breakout-v4",  # Replace 'Breakout-v5' with another game if desired
        "id": atari_game_id,  # Replace 'Breakout-v5' with another game if desired
        "render_mode": "rgb_array",  # None, "human", "rgb_array"
        "full_action_space": False,
        "repeat_action_probability": 0,
        "mode": 0
    }
    env = gym.make(**environment_config)
    # env = AtariWrapper(env)
    env = EpisodicLifeEnv(env)
    env = Monitor(env)
    return env


# Train Stable-Baselines3 Models
def train_stable_baselines(model_class, policy, env, total_timesteps=100000,
                           device="cpu", callbacks=None, model_path="model", log_dir="log_dir", **kwargs):
    model = model_class(policy, env, device=device, verbose=1, tensorboard_log=log_dir, **kwargs)
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(model_path)
    print(f"{model_class.__name__} Model saved to {model_path}")


class RLBaseClass:
    def __init__(self, model_class, total_timesteps, screen_size, policy="", device="",
                 log_dir=None):
        self.model_class = model_class
        self.total_timesteps = total_timesteps
        self.screen_size = screen_size
        self.policy = policy
        self.device = device
        self.log_dir = log_dir

    def make_env(self, screen_size, num_envs, frame_skip, frame_stack):
        env_fns = [make_monitored_env(rank, self.log_dir, frame_skip, self.screen_size) for rank in range(num_envs)]
        vec_env = SubprocVecEnv(env_fns)
        # Add frame stacking (e.g., stack 4 frames)
        stacked_env = VecFrameStack(vec_env, n_stack=frame_stack)
        return stacked_env


class RLTrainingClass(RLBaseClass):
    def __init__(self, model_class, total_timesteps, screen_size, policy, device,
                 model_path, log_dir=None, callbacks=None):
        super().__init__(model_class, total_timesteps, screen_size, policy, device, log_dir)
        self.model_path = model_path

    def train_stable_baselines(self):
        """
        Objective function for Optuna to optimize PPO hyperparameters.
        """
        # Define hyperparameter search space
        # A2C: Best is trial 15 with value: 1.4.
        # optimized_algorithm_params = {'n_steps': 1792,
        #                               'gamma': 0.91,
        #                               'learning_rate': 2.6490911023054706e-05,
        #                               'ent_coef': 0.0016981972066292253,
        #                               'vf_coef': 0.10403420062758251}
        # optimized_env_params = {'num_envs': 1,
        #                         'frame_skip': 4,
        #                         'frame_stack': 4,
        #                         'screen_size': 84}
        # PPO: Best
        # optimized_algorithm_params = {'gamma': 0.99,
        #                               'learning_rate': 0.0009500479588934589,
        #                               'ent_coef': 1.4354716707657966e-06,
        #                               'vf_coef': 0.8338478524724668,
        #                               'n_steps': 1024}
        # STUDENT: IMPORTANT: THE ENVIRONMENT TRAINING PARAMETERS FOR A NETWORK MUST MATCH
        #            IN BOTH THE TRAINING AND AGENT PLAY PARTS
        optimized_env_params = {'num_envs': 12, 'frame_skip': 4, 'frame_stack': 4, 'screen_size': 84}
        optimized_algorithm_params = {}
        # output_file = f"best_hyperparameters_{algorithm_class}.json"
        # with open(output_file, "r") as f:
        #     loaded_hyperparameters = json.load(f)
        # print("Loaded hyperparameters:", loaded_hyperparameters)

        env = self.make_env(**optimized_env_params)
        # Callbacks for rendering and PyTorch TensorBoard logging
        policy_choice_str = "cnn"
        model = self.model_class(self.policy, env, device=self.device, verbose=1,
                                 tensorboard_log=self.log_dir, **optimized_algorithm_params)
        log_dir_algorithm = str(self.model_class.__name__).lower()
        training_callbacks = [
            RenderCallback(env),
            TensorBoardModelCallback(model, frame_skip=optimized_env_params["frame_skip"]),
            SaveModelCallback(save_freq=save_freq, save_path=save_dir,
                              save_prefix=f"{log_dir_algorithm}_{policy_choice_str}", verbose=1)
        ]
        model.learn(total_timesteps=self.total_timesteps, callback=training_callbacks)
        # Close the environment
        model.save(self.model_path)
        print(f"{self.model_class.__name__} Model saved to {self.model_path}")
        env.close()


class ParameterOptimizer(RLBaseClass):
    def __init__(self, model_class, total_timesteps, screen_size, policy, device, log_dir=None, callbacks=None):
        super().__init__(model_class, total_timesteps, screen_size, policy, device, log_dir)

    def train_stable_baselines_optimized(self, trial):
        """
        Objective function for Optuna to optimize PPO hyperparameters.
        """
        # Define hyperparameter search space
        # STUDENT: IMPORTANT: CHANGE THE PARAMTER RANGES AND PARAMETER VARIABLES VALUES FOR EACH ALGORITHM
        optimized_algorithm_params = {
            # All algorithms
            "gamma": trial.suggest_float("gamma", 0.9, 0.99, step=0.01),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            # A2C, PPO, RecurrentPPO
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
            "n_steps": trial.suggest_int("n_steps", 128, 2048, step=128),
            # PPO, RecurrentPPO, DQN, QRDQN
            # "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            # DQN, QRDQN
            # "tau": 1.0,
            # "buffer_size": 250_000,
            # "learning_starts": 250_000 // 2,
        }
        optimized_env_params = {
            "num_envs": trial.suggest_categorical("num_envs", [1, 4, 8, 12]),
            "frame_skip": trial.suggest_int("frame_skip", 1, 5),
            "frame_stack": trial.suggest_int("frame_stack", 1, 5),
            "screen_size": 84
        }
        env = self.make_env(**optimized_env_params)
        model = self.model_class(self.policy, env, device=self.device, verbose=1,
                                 tensorboard_log=self.log_dir, **optimized_algorithm_params)
        training_callbacks = [
            TensorBoardModelCallback(model, frame_skip=optimized_env_params["frame_skip"]),
        ]
        model.learn(total_timesteps=self.total_timesteps, callback=training_callbacks)
        # Evaluate the model
        episode_rewards, episode_lengths = evaluate_policy(model, env, return_episode_rewards=True,
                                                           n_eval_episodes=5, deterministic=True)
        episode_frame_lengths = [episode_length * optimized_env_params["frame_skip"]
                                 for episode_length in episode_lengths]
        # Compute statistics for rewards and lengths
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        std_reward = (sum((x - mean_reward) ** 2 for x in episode_rewards) / len(episode_rewards)) ** 0.5
        mean_frame_length = sum(episode_frame_lengths) / len(episode_frame_lengths)
        std_frame_length = (sum((x - mean_frame_length) ** 2 for x in episode_frame_lengths) / len(
            episode_frame_lengths)) ** 0.5

        print(f"Mean reward: {mean_reward:.2f}, Std dev: {std_reward:.2f}")
        print(f"Mean episode frame length: {mean_frame_length:.2f}, Std dev: {std_frame_length:.2f}")
        # Close the environment
        env.close()
        # Either return mean_reward or mean_frame_length or a normalized combination of the two
        return mean_frame_length


# Function to accumulate key presses in a separate thread
def listen_for_key_presses(key_queue):
    """ Thread to listen for key presses and store them in a queue """
    while not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Quit if 'q' is pressed
                    pygame.quit()
                    return
                key_queue.put(event.key)  # Add the pressed key to the queue
            elif event.type == pygame.KEYUP:
                if event.key in key_queue.queue:
                    key_queue.queue.remove(event.key)  # Remove key when released


# Function for human play
def human_play(env, width=400, height=300, frame_rate=10):
    print("Starting Human Play Mode!")
    print("Controls: Left Arrow = Move Left | Right Arrow = Move Right | Space = Fire")
    print("Press 'q' to quit during gameplay.")

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height))  # Create a small pygame window
    pygame.display.set_caption(f"Human Play: {atari_game_id}")

    clock = pygame.time.Clock()  # Initialize the clock for frame rate control
    font = pygame.font.Font(None, 74)  # Set up a font for the countdown text
    font_small = pygame.font.Font(None, 36)  # Set up a font for the countdown text

    def show_countdown():
        """Displays a 5-second countdown before the game starts."""
        for i in range(2, 0, -1):  # Countdown from 5 to 1
            screen.fill((0, 0, 0))  # Clear the screen
            countdown_text = font.render(str(i), True, (255, 255, 255))  # Render the countdown number
            text_rect = countdown_text.get_rect(center=(width // 2, height // 2))  # Center the text
            screen.blit(countdown_text, text_rect)  # Display the countdown number
            pygame.display.flip()  # Update the screen
            clock.tick(1)  # Wait for 1 second

    def ask_to_play_again(screen, clock):
        text_surface = font_small.render("Press Enter to play again, or Esc to quit.", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
        while True:
            screen.fill((0, 0, 0))  # Clear screen with black background
            screen.blit(text_surface, text_rect)  # Draw text
            pygame.display.flip()  # Update the display
            if not key_queue.empty():
                key = key_queue.get_nowait()  # Get the first key in the queue
                if key == pygame.K_RETURN:  # Enter key
                    return True
                elif key == pygame.K_ESCAPE:  # Quit on 'Esc' key
                    return False
            clock.tick(100)  # Limit the loop to 100 queries per second

    # Map keyboard input to actions
    # Action meanings: [NOOP, FIRE, RIGHT, LEFT]
    action_map = {
        pygame.K_LEFT: 3,  # Move Left
        pygame.K_RIGHT: 2,  # Move Right
        pygame.K_SPACE: 1,  # Fire
    }

    truncated = False

    # Create a queue to store key presses
    key_queue = queue.Queue()
    # Start the thread to listen for key presses
    key_thread = threading.Thread(target=listen_for_key_presses, args=(key_queue,))
    key_thread.daemon = True  # Ensure the thread exits when the main program exits
    key_thread.start()

    # Game loop
    running = True
    while running:  # Loop to allow replaying
        print("Starting a new game!")
        # Show the countdown before the game starts
        show_countdown()
        obs = env.reset()
        game_over = False
        while not game_over:
            obs_image = env.render()  # Returns the rendered frame as an array
            resized_frame = resize_frame(obs_image, width=width, height=height)
            # Convert obs to pygame surface and blit to screen
            # Render the game screen for the human player to see
            screen.blit(pygame.surfarray.make_surface(resized_frame.transpose((1, 0, 2))), (0, 0))
            # Update the display
            pygame.display.flip()
            action = 0  # Default action (NOOP)
            # start_time = pygame.time.get_ticks()  # Record the time before checking for key events
            # Process events within 30 ms
            # while pygame.time.get_ticks() - start_time < 30:  # Allow 30 ms to read a key
            if not key_queue.empty():
                key = key_queue.get_nowait()  # Get the first key in the queue
                if key in action_map:
                    action = action_map[key]  # Update action based on the key press
                elif key == pygame.K_q:  # Quit on 'q' key
                    game_over = True

            if game_over:
                break
            # print(f"action = {action}")
            # Step the environment
            # obs, reward, terminated, truncated, info = env.step(np.array([action]))
            obs, reward, terminated, info = env.step(np.array([action]))
            game_over = terminated or truncated
            # Limit frame rate to frame_rate FPS
            clock.tick(frame_rate)

        print("Game Over!")
        # Ask the user to play again
        running = ask_to_play_again(screen, clock)

    print("\nStopping daemon thread...")
    stop_event.set()  # Signal the thread to stop
    time.sleep(1)
    env.close()
    pygame.quit()
    print("Human Play Session Ended.")


# Function for agent to play the game
def agent_play(env, model, frame_skip=1, width=500, height=400, frame_rate=10):
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height))  # Create a small pygame window
    pygame.display.set_caption(f"RL Agent Play: {atari_game_id}")

    clock = pygame.time.Clock()  # Initialize the clock for frame rate control

    font_small = pygame.font.Font(None, 36)  # Set up a font for the countdown text

    def ask_to_play_again(screen, clock):
        text_surface = font_small.render("Press Enter to play again, or Esc to quit.", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
        while True:
            screen.fill((0, 0, 0))  # Clear screen with black background
            screen.blit(text_surface, text_rect)  # Draw text
            pygame.display.flip()  # Update the display
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:  # Enter key
                        return True
                    elif event.key == pygame.K_ESCAPE:  # Quit on 'Esc' key
                        return False
            clock.tick(100)  # Limit the loop to 100 queries per second

    running = True
    num_frames = 0
    while running:  # Loop to allow replaying
        obs = env.reset()
        done = False
        truncated = False
        num_frames = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            action, _ = model.predict(obs)
            # obs, reward, terminated, truncated, info = env.step(action)
            obs, reward, terminated, info = env.step(action)
            done = terminated or truncated
            obs_image = env.render()  # Returns the rendered frame as an array
            # Rendering and resizing a frame
            resized_frame = resize_frame(obs_image, width=width, height=height)
            # Convert obs to pygame surface and blit to screen
            # Render the game screen for the human player to see
            screen.blit(pygame.surfarray.make_surface(resized_frame.transpose((1, 0, 2))), (0, 0))
            # Update the display
            pygame.display.flip()
            # Limit frame rate to 30 FPS
            clock.tick(frame_rate)
            num_frames += frame_skip

        print(f"Agent played for the equivalent of {num_frames} frames.")
        # Ask the user to play again
        running = ask_to_play_again(screen, clock)
        time.sleep(1)

    env.close()
    print(f"Agent Play Session Ended. Agent played for {num_frames} frames.")


def main(time_steps, training_device, save_dir, save_freq):
    # num_envs_train = 12
    num_envs_play = 1
    # frame_skip = 4
    # frame_stack = 1 * frame_skip
    screen_size = 84

    os.makedirs(save_dir, exist_ok=True)

    # User input for mode selection
    choice = input(
        "Choose mode:\n"
        "1. Train Agent\n"
        "2. Play as Human\n"
        "3. Load Trained Model and Show Agent Playing\n"
        "4. Find Optimal Environment and RL Algorithm Training Parameters\n"
        "Enter your choice: "
    )

    if choice == "1":
        policy_choice = "CnnPolicy"
        policy_choice_str = "cnn"

        algorithm_class = select_model_train()
        log_dir_algorithm = algorithm_class.lower()
        log_dir = os.path.join(".","logs",f"{log_dir_algorithm}_{policy_choice_str}_atari")
        # Use eval to instantiate the class
        model_class = eval(algorithm_class)
        model_path = f"{log_dir_algorithm}_{policy_choice_str}"
        os.makedirs(log_dir, exist_ok=True)

        rl_trainer = RLTrainingClass(model_class, time_steps, screen_size,
                                     policy_choice, training_device, model_path, log_dir=log_dir)
        rl_trainer.train_stable_baselines()

    elif choice == "2":
        # Create a vectorized environment with frame stacking
        env = DummyVecEnv([lambda: make_env(frame_skip=1)])
        human_play(env, width=800, height=600, frame_rate=15)

    elif choice == "3":
        # Create a vectorized environment with frame stacking
        # Replace with your folder path to saved RL models
        selected_model_path = list_and_select_model(save_dir)
        if not selected_model_path:
            return
        # filename = selected_model_path.split("/")[-1]
        filename = os.path.basename(selected_model_path)
        algorithm_class = str(filename.split("_")[0]).upper()
        print(f"Algorithm class name: {algorithm_class}")
        try:
            # Use eval to instantiate the class
            model = eval(algorithm_class).load(selected_model_path)
            print(f"Loading trained model from {selected_model_path}...")
            print("Model loaded. Agent will now play.")
        except NameError:
            print(f"Class '{algorithm_class}' not found!")
            return
        log_dir = None
        # env_fns = [make_monitored_env(rank, log_dir, frame_skip, screen_size) for rank in range(num_envs_play)]
        # vec_env = SubprocVecEnv(env_fns)
        # # Add frame stacking (e.g., stack 4 frames)
        # stacked_env = VecFrameStack(vec_env, n_stack=frame_stack)
        # env = stacked_env
        # STUDENT: IMPORTANT: YOU MUST USE THE SAME ENVIRONMENT PARAMETERS TRAINING AND TESTING/PLAYING THE GAME
        optimized_env_params = {'num_envs': 12, 'frame_skip': 2, 'frame_stack': 4, 'screen_size': 84}

        rl_base = RLBaseClass(algorithm_class, 0, screen_size)
        env = rl_base.make_env(screen_size, num_envs_play,
                               optimized_env_params["frame_skip"],
                               optimized_env_params["frame_stack"])
        agent_play(env, model, frame_skip=optimized_env_params["frame_skip"])

    elif choice == "4":
        log_dir = None
        algorithm_class = select_model_train()
        # Use eval to instantiate the class
        model_class = eval(algorithm_class)
        policy_choice = "CnnPolicy"
        # STUDENT: IMPORTANT: CONSIDER CHANING total_timesteps, n_trials, n_jobs for your project
        total_timesteps = 100_000
        n_trials = 50
        n_jobs = 4
        param_optimizer = ParameterOptimizer(model_class, total_timesteps, screen_size,
                                             policy_choice, training_device, log_dir="./neptuna_logs")
        # Create an Optuna study
        from optuna.integration import PyTorchLightningPruningCallback
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        # study = optuna.create_study(direction="maximize")
        study.optimize(param_optimizer.train_stable_baselines_optimized, n_trials=n_trials, n_jobs=n_jobs)
        # Print a summary of the study
        print("\nStudy Summary:")
        print(f"  Number of trials: {len(study.trials)}")
        print(f"  Best value: {study.best_value}")
        print("Best hyperparameters:", study.best_params)
        # Save the hyperparameters to a JSON file
        output_file = f"best_hyperparameters_{algorithm_class}.json"
        with open(output_file, "w") as f:
            json.dump(study.best_params, f, indent=4)
        print(f"Best hyperparameters saved to {output_file}")
    else:
        print("Invalid choice. Exiting.")


if __name__ == '__main__':
    # atari_game_id = "ALE/Breakout-v5"
    atari_game_id = "ALE/SpaceInvaders-v5"
    # atari_game_id = "ALE/Pong-v5"
    # atari_game_id = "Breakout-v4"
    # Define save frequency (e.g., 100 epochs with 10,000 steps per epoch = 1,000,000 steps)
    save_freq = 100_000  # Adjust this value for your needs
    # Train the model
    time_steps = 1_000_000  # Adjust training steps as needed
    # Save the trained model
    # Define a pattern for invalid characters
    invalid_chars = r'[<>:"/\\|?*]'
    replacement = "_"
    sanitized_id = re.sub(invalid_chars, replacement, atari_game_id)
    game_string = f"atari_{sanitized_id}"
    save_dir = os.path.join(".","models",game_string)

    # Select cpu or gpu, generally gpu will train faster if GPU hardware is available
    # Choose one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy,
    #               vulkan, mps, meta, hpu, mtia, private
    training_device = "cuda"
    # training_device = "cpu"

    gym.register_envs(ale_py)
    main(time_steps, training_device, save_dir, save_freq)
