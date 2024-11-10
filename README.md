# RFAS (Roblox First AI System)

RFAS (Roblox First AI System) is an AI-driven system that leverages reinforcement learning to control a car in a Roblox simulation environment. Built with PyTorch and Flask, RFAS allows users to train AI models, interact with a real-time car simulation, and continuously refine the model based on performance and reward feedback.

## Features
- **AI-powered throttle and steering control** based on state input.
- **Reinforcement learning** with PPO (Proximal Policy Optimization).
- **Real-time interaction** between Roblox and Python using Flask API.
- **Adaptive training** with reward-based feedback to optimize performance.

## Requirements

### Python Dependencies
- Python 3.7 or above
- PyTorch
- Flask
- json
- threading

You can install the dependencies using:
```pip install torch flask```

### Roblox Setup

A Roblox car simulation environment is required, configured to send state data (e.g., speed, throttle, steer, distance) to the Flask API and receive control commands (throttle, steer) from the AI model.

Installation and Setup
----------------------

### Clone the Repository

`git clone https://github.com/yourusername/RFAS.git
cd RFAS`

### Start the Flask Server

Run the following command to start the Flask API server, which listens for incoming data from the Roblox environment and returns AI control commands.


`python Train.py`

### Set Up the Roblox Environment

In Roblox Studio, set up a script to:

1.  Collect car data (distance, speed, steer, throttle) and send it to the Flask API endpoint `/state`.
2.  Retrieve control commands from the API and apply them to the car's Throttle and Steer properties.

Training the AI Model
---------------------

-   The model is configured for continuous training using **PPO (Proximal Policy Optimization)**.
-   Rewards are calculated based on car performance (e.g., maintaining optimal speed, avoiding collisions).
-   The model adjusts throttle and steer commands based on current and previous states to optimize lap completion time.

Usage
-----

### Running the Simulation

1.  **Start the Flask API Server**\
    This server will handle communication between Roblox and the AI model.

    `python Train.py`

2.  **Run the Roblox Simulation**\
    Start your Roblox simulation. The environment script will:

    -   Send the car's state to the Flask API at regular intervals.
    -   Receive throttle and steer values from the AI model in response.
    -   Apply these values to the car in real-time.

### View Logs

-   Training and control information is logged to the console.
-   Use these logs to monitor AI performance and adjust parameters if necessary.
