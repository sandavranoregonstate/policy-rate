import torch
import torch.nn as nn
import torch.optim as optim
import random 
import time 

import game_module as game_module # Make sure to change this according to where your agent file is, mine is currently in the main repository. 

# From the state to all of the actions. 

# TODO: I might be missing no grad. 

class Network( nn.Module ) : 

    def __init__( self, state_dim , action_dim , architecture ) : 
        super( Network , self ).__init__() 
        # Initialize the Neural Network. 
        # TODO: This architecture is random. 

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ) 

    def forward(self, x): 
        # This is required by PyTorch. 
        return self.net(x)

class DQ_agent:
    
    def __init__(self, alpha=0.1, gamma=1, epsilon=0.1 , architecture =1): 
        
        self.model = Network(19, 44, architecture)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha) # TODO: Not sure. (gamma) 
        self.criterion = nn.MSELoss() 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.batch_size = 1000 
        self.trial_count = 1000 
        self.batch_count = 100 
        self.start_time = None 
        self.end_time = None 
        self.nn_batch_size = 500 

        self.possible_if_there_is_remaining_rolls = [ 

            [ 0 , 0 , 0 , 0 , 0 , -1 ] , # roll all 
            [ 0 , 0 , 0 , 0 , 1 , -1 ] , 
            [ 0 , 0 , 0 , 1 , 0 , - 1 ] , 
            [ 0 , 0 , 0 , 1 , 1 , - 1 ] , 
            [ 0 , 0 , 1 , 0 , 0 , - 1 ] , 
            [ 0 , 0 , 1 , 0 , 1 , - 1 ] , 
            [ 0 , 0 , 1 , 1 , 0 , - 1 ] , 
            [ 0 , 0 , 1 , 1 , 1 , - 1 ] , 
            [ 0 , 1 , 0 , 0 , 0 , - 1 ] , 
            [ 0 , 1 , 0 , 0 , 1 , - 1 ] , 
            [ 0 , 1 , 0 , 1 , 0 , - 1 ] , 
            [ 0 , 1 , 0 , 1 , 1 , - 1 ] , 
            [ 0 , 1 , 1 , 0 , 0 , - 1 ] , 
            [ 0 , 1 , 1 , 0 , 1 , - 1 ] , 
            [ 0 , 1 , 1 , 1 , 0 , - 1 ] , 
            [ 0 , 1 , 1 , 1 , 1 , - 1 ] , 
            [ 1 , 0 , 0 , 0 , 0 , - 1 ] , 
            [ 1 , 0 , 0 , 0 , 1 , - 1 ] , 
            [ 1 , 0 , 0 , 1 , 0 , - 1 ] , 
            [ 1 , 0 , 0 , 1 , 1 , - 1 ] , 
            [ 1 , 0 , 1 , 0 , 0 , - 1 ] , 
            [ 1 , 0 , 1 , 0 , 1 , - 1 ] , 
            [ 1 , 0 , 1 , 1 , 0 , - 1 ] , 
            [ 1 , 0 , 1 , 1 , 1 , - 1 ] , 
            [ 1 , 1 , 0 , 0 , 0 , - 1 ] , 
            [ 1 , 1 , 0 , 0 , 1 , - 1 ] , 
            [ 1 , 1 , 0 , 1 , 0 , - 1 ] , 
            [ 1 , 1 , 0 , 1 , 1 , - 1 ] , 
            [ 1 , 1 , 1 , 0 , 0 , - 1 ] , 
            [ 1 , 1 , 1 , 0 , 1 , - 1 ] , 
            [ 1 , 1 , 1 , 1 , 0 , - 1 ] , 

        ] 

        self.action_to_id = { 

            ( 0 , 0 , 0 , 0 , 0,  - 1 ) : 0 , 
            ( 0 , 0 , 0 , 0 , 1,  - 1 ) : 1 , 
            ( 0 , 0 , 0 , 1 , 0,  - 1 ) : 2 , 
            ( 0 , 0 , 1 , 0 , 0,  - 1 ) : 3 , 
            ( 0 , 0 , 1 , 0 , 1,  - 1 ) : 4 , 
            ( 0 , 0 , 1 , 1 , 0,  - 1 ) : 5 , 
            ( 0 , 0 , 1 , 1 , 1,  - 1 ) : 6 , 
            ( 0 , 1 , 0 , 0 , 0,  - 1 ) : 7 , 
            ( 0 , 1 , 0 , 0 , 1,  - 1 ) : 8 , 
            ( 0 , 1 , 0 , 1 , 0,  - 1 ) : 9 , 
            ( 0 , 1 , 0 , 1 , 1,  - 1 ) : 10 , 
            ( 0 , 1 , 1 , 0 , 0,  - 1 ) : 11 , 
            ( 0 , 1 , 1 , 0 , 1,  - 1 ) : 12 , 
            ( 0 , 1 , 1 , 1 , 0,  - 1 ) : 13 , 
            ( 0 , 1 , 1 , 1 , 1,  - 1 ) : 14 , 
            ( 1 , 0 , 0 , 0 , 0,  - 1 ) : 15 , 
            ( 1 , 0 , 0 , 0 , 1,  - 1 ) : 16 , 
            ( 1 , 0 , 0 , 1 , 0,  - 1 ) : 17 , 
            ( 1 , 0 , 0 , 1 , 1,  - 1 ) : 18 , 
            ( 1 , 0 , 1 , 0 , 0,  - 1 ) : 19 , 
            ( 1 , 0 , 1 , 0 , 1,  - 1 ) : 20 , 
            ( 1 , 0 , 1 , 1 , 0,  - 1 ) : 21 , 
            ( 1 , 0 , 1 , 1 , 1,  - 1 ) : 22 , 
            ( 1 , 1 , 0 , 0 , 0,  - 1 ) : 23 , 
            ( 1 , 1 , 0 , 0 , 1,  - 1 ) : 24 , 
            ( 1 , 1 , 0 , 1 , 0,  - 1 ) : 25 , 
            ( 1 , 1 , 0 , 1 , 1,  - 1 ) : 26 , 
            ( 1 , 1 , 1 , 0 , 0,  - 1 ) : 27 , 
            ( 1 , 1 , 1 , 0 , 1,  - 1 ) : 28 , 
            ( 1 , 1 , 1 , 1 , 0,  - 1 ) : 29 , 
            ( 1 , 1 , 1 , 1 , 1,  0 ) : 30 , 
            ( 1 , 1 , 1 , 1 , 1,  1 ) : 31 , 
            ( 1 , 1 , 1 , 1 , 1,  2 ) : 32 , 
            ( 1 , 1 , 1 , 1 , 1,  3 ) : 33 , 
            ( 1 , 1 , 1 , 1 , 1,  4 ) : 34 , 
            ( 1 , 1 , 1 , 1 , 1,  5 ) : 35 , 
            ( 1 , 1 , 1 , 1 , 1,  6 ) : 36 , 
            ( 1 , 1 , 1 , 1 , 1,  7 ) : 37 , 
            ( 1 , 1 , 1 , 1 , 1,  8 ) : 38 , 
            ( 1 , 1 , 1 , 1 , 1,  9 ) : 39 , 
            ( 1 , 1 , 1 , 1 , 1,  10 ) : 40 , 
            ( 1 , 1 , 1 , 1 , 1,  11 ) : 41 , 
            ( 1 , 1 , 1 , 1 , 1,  12 ) : 42 , 
            ( 0 , 0 , 0 , 1 , 1,  -1 ) : 43 

        } 

        self.id_to_action = { # TODO: Change this to list. 

            0 : ( 0 , 0 , 0 , 0 , 0,  - 1 ) , 
            1 : ( 0 , 0 , 0 , 0 , 1,  - 1 ) , 
            2 : ( 0 , 0 , 0 , 1 , 0,  - 1 ) , 
            3 : ( 0 , 0 , 1 , 0 , 0,  - 1 ) , 
            4 : ( 0 , 0 , 1 , 0 , 1,  - 1 ) , 
            5 : ( 0 , 0 , 1 , 1 , 0,  - 1 ) , 
            6 : ( 0 , 0 , 1 , 1 , 1,  - 1 ) , 
            7 : ( 0 , 1 , 0 , 0 , 0,  - 1 ) , 
            8 : ( 0 , 1 , 0 , 0 , 1,  - 1 ) , 
            9 : ( 0 , 1 , 0 , 1 , 0,  - 1 ) , 
            10 : ( 0 , 1 , 0 , 1 , 1,  - 1 ) , 
            11 : ( 0 , 1 , 1 , 0 , 0,  - 1 ) , 
            12 : ( 0 , 1 , 1 , 0 , 1,  - 1 ) , 
            13 : ( 0 , 1 , 1 , 1 , 0,  - 1 ) , 
            14 : ( 0 , 1 , 1 , 1 , 1,  - 1 ) , 
            15 : ( 1 , 0 , 0 , 0 , 0,  - 1 ) , 
            16 : ( 1 , 0 , 0 , 0 , 1,  - 1 ) , 
            17 : ( 1 , 0 , 0 , 1 , 0,  - 1 ) , 
            18 : ( 1 , 0 , 0 , 1 , 1,  - 1 ) , 
            19 : ( 1 , 0 , 1 , 0 , 0,  - 1 ) , 
            20 : ( 1 , 0 , 1 , 0 , 1,  - 1 ) , 
            21 : ( 1 , 0 , 1 , 1 , 0,  - 1 ) , 
            22 : ( 1 , 0 , 1 , 1 , 1,  - 1 ) , 
            23 : ( 1 , 1 , 0 , 0 , 0,  - 1 ) , 
            24 : ( 1 , 1 , 0 , 0 , 1,  - 1 ) , 
            25 : ( 1 , 1 , 0 , 1 , 0,  - 1 ) , 
            26 : ( 1 , 1 , 0 , 1 , 1,  - 1 ) , 
            27 : ( 1 , 1 , 1 , 0 , 0,  - 1 ) , 
            28 : ( 1 , 1 , 1 , 0 , 1,  - 1 ) , 
            29 : ( 1 , 1 , 1 , 1 , 0,  - 1 ) , 
            30 : ( 1 , 1 , 1 , 1 , 1,  0 ) , 
            31 : ( 1 , 1 , 1 , 1 , 1,  1 ) , 
            32 : ( 1 , 1 , 1 , 1 , 1,  2 ) , 
            33 : ( 1 , 1 , 1 , 1 , 1,  3 ) , 
            34 : ( 1 , 1 , 1 , 1 , 1,  4 ) , 
            35 : ( 1 , 1 , 1 , 1 , 1,  5 ) , 
            36 : ( 1 , 1 , 1 , 1 , 1,  6 ) , 
            37 : ( 1 , 1 , 1 , 1 , 1,  7 ) , 
            38 : ( 1 , 1 , 1 , 1 , 1,  8 ) , 
            39 : ( 1 , 1 , 1 , 1 , 1,  9 ) , 
            40 : ( 1 , 1 , 1 , 1 , 1,  10 ) , 
            41 : ( 1 , 1 , 1 , 1 , 1,  11 ) , 
            42 : ( 1 , 1 , 1 , 1 , 1,  12 ) , 
            43 : ( 0 , 0 , 0 , 1 , 1,  -1 ) 

        } 

    def get_the_possible_actions( self , state ) : 
        """ 
        Input: 
        1. State. (a list of length 19) 
        Output: 
        1. A list of possible actions. 
        """

        possible_actions = [] 
        remaining_rolls = state[ 5 ] 
        if remaining_rolls != 0 : 
            for action in self.possible_if_there_is_remaining_rolls : 
                possible_actions.append( action ) 
        remaining_categories = [] 
        for idx in range( 13 ) : 
            if state[ idx + 6 ] != 1 : 
                remaining_categories.append( idx ) 
        for category_idx in remaining_categories : 
            possible_actions.append( [ 1 , 1 , 1 , 1 , 1 , category_idx ] ) 
        return possible_actions 
    
    def get_epsilon_greedy_action( self , state ) : 
        """
        Input: 
        1. State. (a list of length 19) 

        Output: 
        1. Action. (a list of length 6) 

        To Do: 
        1. Get the possible actions. ([action]) (this is not an integer)
        2. Get the q values. 
        3. Figure out the id of each action. 
        4. Use the algorithm. 
        """ 
        # 1. 
        possible_actions = self.get_the_possible_actions(state) 
        # 2. 
        state = torch.tensor(state, dtype=torch.float32) 
        q_value_list = self.model(state) 
        # 3. 
        valid_set = set() 
        for action in possible_actions : 
            valid_set.add( self.action_to_id[ tuple( action ) ] ) 
        # 4. 
        best_q = float("-inf") # TODO: Not sure. 
        best_action = None 
        for idx in range( len( q_value_list ) ) : 
            if q_value_list[ idx ] > best_q and idx in valid_set : 
                best_q = q_value_list[ idx ]
                best_action = self.id_to_action[ idx ]  
        if random.random() > self.epsilon: 
            return best_action 
        else: 
            return random.choice(possible_actions) 

    def update_model(self, state_list, action_list, reward_list, next_state_list, is_terminal_list):
        """
        Input: 
        1. State List. 
        2. Action List. 
        3. Reward List. 
        4. Next State List. 
        5. Is Terminal List. 

        Output: 
        1. None. 
        """ 
        # Regular to PyTorch. 
        state_list = torch.tensor( state_list , dtype=torch.float32 ) 
        next_state_list = torch.tensor( next_state_list , dtype=torch.float32 ) 
        reward_list = torch.tensor( reward_list , dtype=torch.float32 ) 
        is_terminal_list = torch.tensor( is_terminal_list , dtype=torch.bool ) 
        # Use the id of the action. 
        action_list = torch.tensor( [ self.action_to_id[ tuple( action ) ] for action in action_list], dtype = torch.int64 ) 
        # Get certain q value. 
        q_values = self.model( state_list ) # Return nn_batch_size X action_space_size. 
        next_q_values = self.model( next_state_list ) # # Return nn_batch_size X action_space_size. 
        # Find the q value of the action agent has taken. 
        q_value = q_values.gather( 1 , action_list.unsqueeze( 1 ) ).squeeze( 1 ) 
        # Calculate the target Q-values
        next_q_value = next_q_values.max( 1 )[ 0 ] 
        expected_q_value = reward_list + self.gamma * next_q_value * ( 1 - is_terminal_list.to( torch.float32 ) ) # This is weird from the textbook, if it is the final state, don't check it. 
        # Calculate the loss
        loss = self.criterion( q_value , expected_q_value ) 
        # Backpropagate and update the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 

    def train( self ) :  
        
        state_list = [] 
        action_list = [] 
        reward_list = [] 
        next_state_list = [] 
        is_terminal_list = [] 
        # Loop each episode. 
        for episode_id in range( self.batch_size ) : 
            # print(episode_id)
            game = game_module.Game() 
            state = game.reset()
            is_terminal = False
            # Loop for each step of the episode. 
            while is_terminal == False : 
                action = self.get_epsilon_greedy_action( state ) 
                next_state , reward , is_terminal = game.step( action ) 
                # Add it to the list. 
                state_list.append( state ) 
                action_list.append(action) 
                reward_list.append(reward) 
                next_state_list.append(next_state) 
                is_terminal_list.append(is_terminal) 
                state = next_state 
            
            if ( episode_id + 1 ) % self.nn_batch_size == 0 : 
                # print("update_batch")
                self.update_model( state_list , action_list , reward_list , next_state_list , is_terminal_list ) 
                state_list = [] 
                action_list = [] 
                reward_list = [] 
                next_state_list = [] 
                is_terminal_list = [] 

    def test( self ) : 
        
        sum_reward = 0 
        # Loop each trial count. 
        for _ in range( self.trial_count ) : 
            game = game_module.Game() 
            state  = game.reset() 
            is_terminal = False 
            # Loop for each step of the episode. 
            while is_terminal == False : 
                action = self.get_epsilon_greedy_action( state ) 
                next_state , reward , is_terminal = game.step( action ) 
                state = next_state 
                sum_reward += reward # TODO: might not be correct. 
        
        return sum_reward / self.trial_count 
    
    def train_test( self ) : 

        self.start_time = time.time() 

        for _ in range( self.batch_count ) : 
            # print(f"test_count={test_count}")
            self.train() 
            average_reward = self.test() 
            self.end_time = time.time() 
            print( average_reward  , ", " , self.end_time - self.start_time ) 
            # self.reward_list.append( average_reward ) 

import sys 

epsilon = float(sys.argv[1]) 
alpha = float(sys.argv[2]) 
architecture = float(sys.argv[3])

print(1)
agent = DQ_agent( epsilon , alpha , architecture ) 
agent.train_test() 

