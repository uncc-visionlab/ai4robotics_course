import gymnasium as gym
import ale_py
import cv2
from stable_baselines3.common.atari_wrappers import AtariWrapper, EpisodicLifeEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pygame
import queue
import re
import sys
import threading
import time


def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


# # Initialize Pygame
# pygame.init()
#
# # Constants
# SCREEN_WIDTH, SCREEN_HEIGHT = 400, 400
# BG_COLOR = (0, 0, 0)
# FPS = 30
#
# # Create Pygame screen
# screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# pygame.display.set_caption("Atari RL Keyboard Handler")

# Clock for controlling frame rate
clock = pygame.time.Clock()

# Define Atari RL action mapping based on the stable-baselines3 default
# Add other Atari games actions as needed
ATARI_ACTIONS = {
    "NOOP": 0,  # No action
    "FIRE": 1,  # Fire button only
    "UP": 2,  # Move
    "RIGHT": 3,  # Move
    "LEFT": 4,  # Move
    "DOWN": 5,  # Move
    "UPRIGHT": 6,  # Move
    "UPLEFT": 7,  # Move
    "DOWNRIGHT": 8,  # Move
    "DOWNLEFT": 9,  # Move
    "UPFIRE": 10,  # Move + fire
    "RIGHTFIRE": 11,  # Move + fire
    "LEFTFIRE": 12,  # Move + fire
    "DOWNFIRE": 13,  # Move + fire
    "UPRIGHTFIRE": 14,  # Move + fire
    "UPLEFTFIRE": 15,  # Move + fire
    "DOWNRIGHTFIRE": 16,  # Move + fire
    "DOWNLEFTFIRE": 17,  # Move + fire
    "RESET": 40
}

# Map each movement to its corresponding RL action value
RL_ACTIONS = {
    "NOOP": ATARI_ACTIONS["NOOP"],
    "FIRE": ATARI_ACTIONS["FIRE"],
    "UP": ATARI_ACTIONS["UP"],  # Move
    "RIGHT": ATARI_ACTIONS["RIGHT"],  # Move
    "LEFT": ATARI_ACTIONS["LEFT"],  # Move
    "DOWN": ATARI_ACTIONS["DOWN"],  # Move
    "UPRIGHT": ATARI_ACTIONS["UPRIGHT"],  # Move
    "UPLEFT": ATARI_ACTIONS["UPLEFT"],  # Move
    "DOWNRIGHT": ATARI_ACTIONS["DOWNRIGHT"],  # Move
    "DOWNLEFT": ATARI_ACTIONS["DOWNLEFT"],  # Move
    "UPFIRE": ATARI_ACTIONS["UPFIRE"],  # Move + fire
    "RIGHTFIRE": ATARI_ACTIONS["RIGHTFIRE"],  # Move + fire
    "LEFTFIRE": ATARI_ACTIONS["LEFTFIRE"],  # Move + fire
    "DOWNFIRE": ATARI_ACTIONS["DOWNFIRE"],  # Move + fire
    "UPRIGHTFIRE": ATARI_ACTIONS["UPRIGHTFIRE"],  # Move + fire
    "UPLEFTFIRE": ATARI_ACTIONS["UPLEFTFIRE"],  # Move + fire
    "DOWNRIGHTFIRE": ATARI_ACTIONS["DOWNRIGHTFIRE"],  # Move + fire
    "DOWNLEFTFIRE": ATARI_ACTIONS["DOWNLEFTFIRE"],  # Move + fire
    "RESET": ATARI_ACTIONS["RESET"]
}

# Define movement and shift key mappings
KEYPAD_MOVEMENTS = {
    pygame.K_KP1: "DOWNLEFT",
    pygame.K_KP2: "DOWN",
    pygame.K_KP3: "DOWNRIGHT",
    pygame.K_KP4: "LEFT",
    pygame.K_KP5: "NOOP",
    pygame.K_KP6: "RIGHT",
    pygame.K_KP7: "UPLEFT",
    pygame.K_KP8: "UP",
    pygame.K_KP9: "UPRIGHT",
}
# KEYPAD_MOVEMENTS = {
#     pygame.K_z: "DOWNLEFT",
#     pygame.K_x: "DOWN",
#     pygame.K_c: "DOWNRIGHT",
#     pygame.K_a: "LEFT",
#     pygame.K_s: "NOOP",
#     pygame.K_d: "RIGHT",
#     pygame.K_q: "UPLEFT",
#     pygame.K_w: "UP",
#     pygame.K_e: "UPRIGHT",
# }

# Add shift-modified actions
SHIFT_MOVEMENTS = {key: f"{action}FIRE" for key, action in KEYPAD_MOVEMENTS.items()}

# Shared flag for stopping the thread
stop_event = threading.Event()


# Function to accumulate key presses in a separate thread
def listen_for_key_presses(action_queue):
    clock = pygame.time.Clock()  # Initialize the clock for frame rate control
    """ Thread to listen for key presses and store them in a queue """
    while not stop_event.is_set():
        for event in pygame.event.get():
            keys = pygame.key.get_pressed()
            shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

            # Default action is noop
            selected_action = RL_ACTIONS["NOOP"]

            # Check for keypad movements
            for key, action in KEYPAD_MOVEMENTS.items():
                if keys[key]:
                    if keys[key] == pygame.K_q:
                        pygame.quit()
                        return
                    elif keys[key] == pygame.K_ESCAPE:
                        selected_action = RL_ACTIONS["RESET"]
                    elif shift_pressed and action in SHIFT_MOVEMENTS.values():
                        selected_action = RL_ACTIONS.get(f"{action}FIRE", RL_ACTIONS["NOOP"])
                    else:
                        selected_action = RL_ACTIONS.get(action, RL_ACTIONS["NOOP"])

            # If only shift is pressed, assign the fire action
            if shift_pressed and selected_action == RL_ACTIONS["NOOP"]:
                selected_action = RL_ACTIONS["FIRE"]

            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                action_queue.put(selected_action)  # Add the pressed key to the queue
            elif event.type == pygame.KEYUP:
                # if event.key in key_queue.queue:
                action_queue.clear()
            clock.tick(300)

# Function to handle keyboard input
def handle_keyboard_input():
    keys = pygame.key.get_pressed()
    shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

    # Default action is noop
    selected_action = RL_ACTIONS["noop"]

    # Check for keypad movements
    for key, action in KEYPAD_MOVEMENTS.items():
        if keys[key]:
            if shift_pressed and action in SHIFT_MOVEMENTS.values():
                selected_action = RL_ACTIONS.get(f"{action}+fire", RL_ACTIONS["noop"])
            else:
                selected_action = RL_ACTIONS.get(action, RL_ACTIONS["noop"])

    # If only shift is pressed, assign the fire action
    if shift_pressed and selected_action == RL_ACTIONS["noop"]:
        selected_action = RL_ACTIONS["fire"]

    return selected_action


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
    # env = Monitor(env)
    return env


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
        text_surface = font_small.render("Press Esc to quit or Any Other Key to Replay.", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
        while True:
            screen.fill((0, 0, 0))  # Clear screen with black background
            screen.blit(text_surface, text_rect)  # Draw text
            pygame.display.flip()  # Update the display
            if not action_queue.empty():
                selected_action = action_queue.get_nowait()  # Get the first key in the queue
                if selected_action == RL_ACTIONS["RESET"]:  # Quit on 'Esc' key
                    return False
                else:  # Else replay
                    return True
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
    action_queue = queue.Queue()
    # Start the thread to listen for key presses
    key_thread = threading.Thread(target=listen_for_key_presses, args=(action_queue,))
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
        # Get the action space
        action_space = env.action_space
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
            if not action_queue.empty():
                action = action_queue.get_nowait()  # Get the first key in the queue
                if not (isinstance(action, int) and 0 <= action < action_space.n):
                    print(f"Invalid action value = {action} returned!")
                    action = 0

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


# # Main loop
# def main():
#     while True:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 sys.exit()
#
#         # Get the action based on keyboard input
#         action = handle_keyboard_input()
#
#         # Update screen
#         screen.fill(BG_COLOR)
#         font = pygame.font.Font(None, 36)
#         text_surface = font.render(f"Selected Action: {action}", True, (255, 255, 255))
#         screen.blit(text_surface, (10, SCREEN_HEIGHT // 2 - 20))
#
#         # Update display
#         pygame.display.flip()
#         clock.tick(FPS)


# Run the program
# if __name__ == "__main__":
#     main()

if __name__ == '__main__':
    # atari_game_id = "ALE/Breakout-v5"
    atari_game_id = "Breakout-v4"
    # Save the trained model
    # Define a pattern for invalid characters
    invalid_chars = r'[<>:"/\\|?*]'
    replacement = "_"
    sanitized_id = re.sub(invalid_chars, replacement, atari_game_id)
    game_string = f"atari_{sanitized_id}"
    gym.register_envs(ale_py)
    env = DummyVecEnv([lambda: make_env(frame_skip=1)])
    human_play(env, width=800, height=600, frame_rate=15)

# Create multiple environments with SubprocVecEnv
# env_fns = [make_monitored_env(rank, log_dir, frame_skip, screen_size) for rank in range(num_envs_train)]
# vec_env = SubprocVecEnv(env_fns)
# Add frame stacking (e.g., stack 4 frames)
# stacked_env = VecFrameStack(vec_env, n_stack=frame_stack)
# env = stacked_env
# print("Starting training...")
# Callbacks for rendering and PyTorch TensorBoard logging
# training_callbacks = [
#     RenderCallback(env),
#     TensorBoardCallback(writer=writer, frame_skip=frame_skip),
#     SaveModelCallback(save_freq=save_freq, save_path=save_dir,
#                       save_prefix=f"{log_dir_algorithm}_{policy_choice_str}", verbose=1)
# ]

# Map algorithm names to their corresponding functions or classes
# algorithm_map = {
#     "REINFORCE": lambda: train_stable_baselines(PPO, policy_choice, env, time_steps,
#                                                 device=training_device, callbacks=training_callbacks,
#                                                 model_path=model_path, log_dir=log_dir),
#     "A2C": lambda: train_stable_baselines(A2C, policy_choice, env, time_steps,
#                                           device=training_device, callbacks=training_callbacks,
#                                           model_path=model_path, log_dir=log_dir),
#     "DQN": lambda: train_stable_baselines(DQN, policy_choice, env, time_steps,
#                                           device=training_device, callbacks=training_callbacks,
#                                           model_path=model_path, log_dir=log_dir,
#                                           buffer_size=dqn_buffer_size,
#                                           learning_starts=dqn_learning_starts),
#     "PPO": lambda: train_stable_baselines(PPO, policy_choice, env, time_steps,
#                                           device=training_device, callbacks=training_callbacks,
#                                           model_path=model_path, log_dir=log_dir),
#     "TRPO": lambda: train_stable_baselines(TRPO, policy_choice, env, time_steps,
#                                            device=training_device, callbacks=training_callbacks,
#                                            model_path=model_path, log_dir=log_dir),
#     "QRDQN": lambda: train_stable_baselines(QRDQN, policy_choice, env, time_steps,
#                                             device=training_device, callbacks=training_callbacks,
#                                             model_path=model_path, log_dir=log_dir,
#                                             buffer_size=dqn_buffer_size,
#                                             learning_starts=dqn_learning_starts),
#     "RPPO": lambda: train_stable_baselines(RecurrentPPO, "CnnLstmPolicy", env, time_steps,
#                                            device=training_device, callbacks=training_callbacks,
#                                            model_path=model_path, log_dir=log_dir),
# }

# Check if the algorithm is valid
# if algorithm_choice in algorithm_map:
#     algorithm_map[algorithm_choice]()  # Call the corresponding training function
# else:
#     print(f"Algorithm {algorithm_map[algorithm_choice]} is not supported.")
