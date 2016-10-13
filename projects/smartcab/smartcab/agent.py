from __future__ import print_function
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.actions = [None, 'forward','right', 'left']
        self.qTable = {}
        self.alpha = 0.5 # learning rate
        self.gamma = 0.5 # discount factor
        self. epsilon = 0.01 # random factor
        self.state_previous = None # previous state
        self.action_previous = None # previous action
        self.reward_previous = None # previous reward value

        self.trial_count = 0 # number of trials finished
        self.last_rewards = 0   # total rewards in last trial
        self.last_actions = 0   # total number of actions in last trial
        self.last_negative_reward_count = 0 # total number of negative reward in last trial
        self.last_rewards_list = {} # list of total rewards in each trial
        self.last_actions_list = {} # list of total number of actions in each trial
        self.last_negative_reward_count_list = {} # list of total number of negative reward in each trial

        self.total_rewards = 0  # all trials' total rewards
        self.total_actions = 0  # all trials' total number of actions

        # initialize Q table
        for light in ['red', 'green']:
            for oncoming in [None, 'forward','right', 'left']:
                for left in [None, 'forward','right', 'left']:
                    for next_waypoint in [None, 'forward','right', 'left']:
                        for action in [None, 'forward','right', 'left']:
                            self.qTable[((light, oncoming, left, next_waypoint), action)] = 0.0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.trial_count += 1
        self.last_negative_reward_count = 0
        self.last_rewards = 0
        self.last_actions = 0


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)


        # TODO: Update state
        state = (inputs['light'], inputs['oncoming'], inputs['left'],
                     self.next_waypoint)
        self.state = state

        """
        # driving agent randomly chooses actions
        action = random.choice(self.actions)
        """

        # TODO: Select action according to your policy
        values = [self.qTable[(state, None)], self.qTable[(state, 'forward')],self.qTable[(state, 'right')], self.qTable[(state, 'left')]]

        if max(values) == 0 or random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = self.actions[np.argmax(values)]

        # Execute action and get reward
        reward = self.env.act(self, action)


        # TODO: Learn policy based on state, action, reward
        if self.total_actions > 0:
            self.qTable[(self.state_previous, self.action_previous)] += self.alpha * (self.reward_previous + (self.gamma * (max(values)) - self.qTable[(self.state_previous, self.action_previous)]))

        self.state_previous = self.state
        self.action_previous = action
        self.reward_previous = reward

        self.total_rewards += reward
        self.last_rewards += reward
        self.total_actions += 1
        self.last_actions += 1
        if reward < 0 :
            self.last_negative_reward_count += 1
        self.last_rewards_list[self.trial_count] = self.last_rewards
        self.last_actions_list[self.trial_count] = self.last_actions
        self.last_negative_reward_count_list[self.trial_count] = self.last_negative_reward_count

        print ("LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward) )# [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline = True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0005, display = False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print ("................................DATA SUMMARY................................")
    print ("alpha: ", a.alpha)
    print ("gamma: ", a.gamma)
    print ("epsilon: ", a.epsilon)
    print ("total actions: ", a.total_actions)
    print ("total rewards: ", a.total_rewards)
    print ("number of negative reward in each trial ", a.last_negative_reward_count_list.values())
    print ("number of actions in each trial ", a.last_actions_list.values())
    print ("total rewards in each trial ", a.last_rewards_list.values())

if __name__ == '__main__':
    run()
