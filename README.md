# gym-snake

This repository contains an OpenAI environment implementation of the [snake video game](https://en.wikipedia.org/wiki/Snake_(video_game_genre)).

## Installation

Install [OpenAI gym](https://gym.openai.com/docs/).

Install this package:

```
cd gym-snake
pip install -e .
```

## Usage

```
import gym
import gym_snake

env = gym.make('Snake-v0', height=5, width=10)
```
