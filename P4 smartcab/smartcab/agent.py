import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
from collections import OrderedDict
import pickle

# to remind myself: Environment.valid_actions = [None, 'forward', 'left', 'right']
# env.sense will return {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
# light has 2 possbile options, the other 3 have 4 possible condtions.
# self.next_waypoint also returns 4 possbile values, which represents which direction one should go in order to reach the destination
# finally, action = self.next_waypoint, perform the chosen action


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Qtable = {}
        self.StateCounter = {}
        self.alpha = 0.9 # learning rate
        self.gamma = 0.3 # discount
        self.epsilon = 0.6 # probability to reject current decision
        self.decay = 0.2 # I choose this value to control speed epsilon decay: epsilon = epsilon/(1+decay*time)
        self.statefilepath = ('StateAppear.txt') # save new state appear in this file
        self.decayValid = True # control whether use decreasing alpha & epsilon or not
        self.previousState = None
        self.previousDecision = None
        self.previousReward = None
        self.previousAlpha = None
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        #self.Qtable = {}
        #self.StateCounter = {}
        #self.state = None
        #self.next_waypoint = None
        self.previousState = None
        self.previousDecision = None
        self.previousReward = None
        self.previousAlpha = None
        
    def Alpha_decay(self,CurrentCounter,valid):
        if valid:
            return self.alpha/float(CurrentCounter)
        else:
            return self.alpha
        
    def Epsilon_decay(self,CurrentCounter,valid):
        if valid:
            return self.epsilon/float(1+self.decay*CurrentCounter)
        else:
            return self.epsilon
        
    def State2txt(self, CurrentState):
        with open(self.statefilepath, 'a') as file:
            message = 'NewStateAppear: {}\n'.format(CurrentState)
            file.write(message)
            file.close()


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        CurrentState = OrderedDict()
        CurrentState['guide'] = self.next_waypoint
        CurrentState['light'] = inputs['light']
        CurrentState['oncoming'] = inputs['oncoming']
        CurrentState['left'] = inputs['left']
        CurrentState['right'] = inputs['right']
        #CurrentState['deadline'] = deadline
        CurrentState = tuple(CurrentState.items()) # use this tuple to represent possible state
        self.state = CurrentState
        
        if not(self.StateCounter.has_key(CurrentState)):
            #self.Qtable[CurrentState] = np.zeros(4)
            self.Qtable[CurrentState] = np.random.rand(4)
            #self.Qtable[CurrentState] = np.array([3,3,3,3])
            self.StateCounter[CurrentState] = 0
            #self.State2txt(CurrentState)
            
        
        # TODO: Select action according to your policy
        #action = None
        CurrentQ = self.Qtable[CurrentState]
        self.StateCounter[CurrentState] += 1
        CurrentCounter = self.StateCounter[CurrentState]
        Decision = CurrentQ.argmax()
        prob = random.uniform(0,1)
        RealDecision = Decision
        
        ### with a probability epsilon, we randomly choose a direction
        if prob > self.Epsilon_decay(CurrentCounter,valid = self.decayValid):
            action = Environment.valid_actions[RealDecision]
        else:
            #self.next_waypoint = random.choice(Environment.valid_actions[1:])
            RealDecision = random.choice([1,2,3])
            action = Environment.valid_actions[RealDecision]

        # Execute action and get reward
        reward = self.env.act(self, action)
        alpha_tmp = self.Alpha_decay(CurrentCounter,valid = self.decayValid)

        # TODO: Learn policy based on state, action, reward
        if not (self.previousState == None):
            #self.Qtable[self.previousState][self.previousDecision] = (1-self.previousAlpha) * self.Qtable[self.previousState][self.previousDecision] + self.previousAlpha * (self.previousReward + self.gamma*self.Qtable[CurrentState].max())
            self.Qtable[self.previousState][self.previousDecision] = (1-alpha_tmp) * self.Qtable[self.previousState][self.previousDecision] + alpha_tmp * (self.previousReward + self.gamma*self.Qtable[CurrentState].max())
            #self.Qtable[self.previousState][self.previousDecision] = (1-alpha_tmp) * self.Qtable[self.previousState][self.previousDecision] + alpha_tmp * (reward + self.gamma*self.Qtable[CurrentState].max())
        
        self.previousState = CurrentState
        self.previousDecision = RealDecision
        self.previousReward = reward
        #self.previousAlpha = alpha_tmp
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    #e.set_primary_agent(a, enforce_deadline=False)
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    #sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=1000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    
    #a.Qtable
    #f=file('data.txt','wb')   
    #pickle.dump(a.Qtable,f)
    #f.close()

if __name__ == '__main__':
    run()
