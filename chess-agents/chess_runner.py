#!/usr/bin/env python3

import gymnasium as gym
import chess
import argparse
import logging
import sys
import time

def create_environment(render_mode, seed=None):
    env = chess.env(render_mode=render_mode)
    if seed:
        env.reset(seed)
    return env

def destroy_environment(env):
    env.close()
    return

def run_one_episode(env, agent1, agent2):
    agents = { 'White': agent1, 'Black': agent2 }
    times = { "White": 0.0, "Black": 0.0 }
    env.reset()
    agent1.reset()
    agent2.reset()
    env.model.render_board(env.model.initialize_board())
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if all(env.terminations.values()):
            print(f"Game ended. Final rewards: {env.rewards}")
            break 
        
        if termination or truncation:
            env.step(None)
            continue

        t1 = time.time()
        action = agents[agent].agent_function(observation,env, agent)
        t2 = time.time()
        print("agent:",agent)
        times[agent] += (t2-t1)
        env.step(action)


    
    winner = None
    rewards = env.rewards
    if rewards.get('White', 0) > rewards.get('Black', 0):
        winner = 'White'
    elif rewards.get('White', 0) < rewards.get('Black', 0):
        winner = 'Black'
    else:
        print("Game ends in a draw.")
        
    # for agent in times:
    #     print(f"{agent} took {times[agent]:8.5f} seconds.")

    return winner

def run_many_episodes(env, episode_count, agent1, agent2):
    winners = {}
    for i in range(episode_count):
        winner = run_one_episode(env, agent1, agent2)
        if winner not in winners:
            winners[winner] = 0
        winners[winner] += 1
    destroy_environment(env)
    return winners

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Run Chess')
    parser.add_argument(
        "--episode-count",
        "-c",
        type=int, 
        help="number of episodes to run",
        default=5
    )
    parser.add_argument(
        "--logging-level",
        "-l",
        type=str,
        help="logging level: warn, info, debug",
        choices=("warn", "info", "debug"),
        default="warn",
    )
    parser.add_argument(
        "--seed",
        type=int, 
        help="seed for the environment's PRNG",
        default=0
    )
    parser.add_argument(
        "--render-mode",
        "-r",
        type=str,
        help="display style (render mode): human, none",
        choices=("human", "none"),
        default="human",
    )
    parser.add_argument(
        "--agent1",
        "-a",
        type=str,
        help="agent function: random, killer",
        choices=("random", "agent1", "killer", "human","alphav1","alphav2","alphav3"),
        default="killer",
    )
    parser.add_argument(
        "--agent2",
        "-A",
        type=str,
        help="agent function: random, killer",
        choices=("random", "agent1", "killer", "human","alphav1", "alphav2","alphav3"),
        default="alphav2",
    )

    my_args = parser.parse_args(argv[1:])
    if my_args.logging_level == "warn":
        my_args.logging_level = logging.WARN
    elif my_args.logging_level == "info":
        my_args.logging_level = logging.INFO
    elif my_args.logging_level == "debug": 
        my_args.logging_level = logging.DEBUG

    if my_args.render_mode == "none":
        my_args.render_mode = None
    return my_args

from random_agent import AgentRandom
from alpha_beta import AgentAlphaBeta as AgentKiller
from alpha_beta1 import AgentAlphaBetav1 as Alphav1
from human import AgentHuman
from alpha_beta2 import AgentAlphaBetav2 as Alphav2
from alpha_beta3 import AgentAlphaBetav3 as Alphav3

def select_agent(agent_name):
    if agent_name == "random": 
        agent_function = AgentRandom()
    elif agent_name == "killer":
        agent_function = AgentKiller()
    elif agent_name == "human":
        agent_function = AgentHuman()
    elif agent_name == "alphav1":
        agent_function = Alphav1()
    elif agent_name == "alphav2":
        agent_function = Alphav2()
    elif agent_name == "alphav3":
        agent_function = Alphav3()
    else:
        raise Exception(f"unknown agent name: {agent_name}")
    return agent_function

def main(argv):
    my_args = parse_args(argv)
    logging.basicConfig(level=my_args.logging_level)

    env = create_environment(my_args.render_mode, my_args.seed)
    agent1 = select_agent(my_args.agent1)
    agent2 = select_agent(my_args.agent2)
    winners = run_many_episodes(env, my_args.episode_count, agent1, agent2)
    print(f"Winners: {winners}")
    return

if __name__ == "__main__":
    main(sys.argv)

