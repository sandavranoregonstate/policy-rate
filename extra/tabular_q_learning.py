import game_module as game_module # Make sure to change this according to where your agent file is, mine is currently in the main repository. 
import random 
import time 

class TQ_agent: 
    
    def __init__( self , epsilon = 0.05 , alpha = 0.05 , batch_size = 2000 , batch_count = 10000 , trial_count = 10000 ) : 

        # 1. Initialise the hyper parameters. 
        self.epsilon = epsilon  
        self.alpha = alpha  
        self.gamma = 1  
        self.batch_size = batch_size 
        self.batch_count = batch_count 
        self.trial_count = trial_count 
        self.reward_list = [] 
        self.q_table = {} # { ( tuple( state ) , tuple( action ) ) : value } 

        self.start_time = None 
        self.end_time = None 

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
    
    def get_epsilon_greedy_action(self, state): # TODO: Not sure. 
        
        possible_actions = self.get_the_possible_actions(state)
        best_q = float("-inf")
        best_action = None
        for action in possible_actions: 
            # Check if the curr_hash is in the table. 
            curr_hash = ( tuple( state ) , tuple( action ) ) 
            if curr_hash not in self.q_table : 
                self.q_table[curr_hash] = 0 
            current_q = self.q_table[curr_hash]
            if current_q > best_q:
                best_q = current_q
                best_action = action
        if random.random() > self.epsilon: 
            return best_action 
        else: 
            return random.choice(possible_actions) 

    def get_the_best_action( self , state ) : 

        possible_actions = self.get_the_possible_actions( state ) 
        best_q = float("-inf") 
        best_action = None 
        for action in possible_actions : 
            # Check if the curr_hash is in the table. 
            curr_hash = ( tuple( state ) , tuple( action ) ) 
            if curr_hash not in self.q_table : 
                # print(f"ping.")
                self.q_table[curr_hash] = 0 
            current_q = self.q_table[curr_hash]
            if current_q > best_q : 
                best_q = current_q 
                best_action = action 
        return best_action 

    def update_the_q_table( self , state , action , reward , next_state , best_action ) : 
        
        if best_action == None : # TODO: this is weird 
            self.q_table[ ( tuple(state) , tuple( action ) ) ] = self.q_table[ ( tuple(state) , tuple( action ) ) ] + self.alpha * ( reward + self.gamma * 0 - self.q_table[ ( tuple(state) , tuple( action ) ) ] )  
        else : 
            if ( tuple(state) , tuple( action ) ) not in self.q_table : 
                self.q_table[ ( tuple(state) , tuple( action ) ) ] = 0 
            if ( tuple( next_state ) , tuple( best_action ) ) not in self.q_table : 
                self.q_table[ ( tuple( next_state ) , tuple( best_action ) ) ] = 0 
            self.q_table[ ( tuple(state) , tuple( action ) ) ] = self.q_table[ ( tuple(state) , tuple( action ) ) ] + self.alpha * ( reward + self.gamma * self.q_table[ ( tuple(next_state) , tuple( best_action ) ) ] - self.q_table[ ( tuple(state) , tuple( action ) ) ] ) 

    def get_the_possible_actions( self , state ) : 

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

    def train( self , just_lower = False ) : 

        # Loop each episode. 
        for _ in range( self.batch_size ) : 
            game = game_module.Game() 
            if just_lower == True : 
                state = game.resetHalf() 
            else : 
                state  = game.reset() 
            is_terminal = False 
            # Loop for each step of the episode. 
            while is_terminal == False : 
                action = self.get_epsilon_greedy_action( state ) 
                # print(f"state={state},action={action}")
                next_state , reward , is_terminal = game.step( action ) 
                best_action = self.get_the_best_action( next_state ) 
                self.update_the_q_table( state , action , reward , next_state , best_action ) 
                state = next_state 

    def test( self , just_lower = False ) : 
        
        sum_reward = 0 
        # Loop each trial count. 
        for _ in range( self.trial_count ) : 
            game = game_module.Game() 
            if just_lower == True : 
                state = game.resetHalf() 
            else : 
                state  = game.reset() 
            is_terminal = False 
            # Loop for each step of the episode. 
            while is_terminal == False : 
                action = self.get_the_best_action( state ) 
                next_state , reward , is_terminal = game.step( action ) 
                state = next_state 
                sum_reward += reward # TODO: might not be correct. 
        
        return sum_reward / self.trial_count 
    
    def just_train( self , just_lower ) : 

        for _ in range( self.batch_count ) : 
            self.train( just_lower )  

    def train_test( self , just_lower = False ) : 
        self.start_time = time.time() 
        for _ in range( self.batch_count ) : 
            self.train( just_lower )  
            average_reward = self.evaluate( self.trial_count , just_lower ) 
            self.end_time = time.time() 
            print( average_reward , "," , self.end_time - self.start_time ) 

    def test_random_agent( self , just_lower ) : 

        sum_reward = 0 
        # Loop each trial count. 
        for _ in range( self.trial_count ) : 
            game = game_module.Game() 
            if just_lower == True : 
                state = game.resetHalf() 
            else : 
                state  = game.reset() 
            is_terminal = False 
            # Loop for each step of the episode. 
            while is_terminal == False : 
                action = self.get_random_action( state ) 
                next_state , reward , is_terminal = game.step( action ) 
                state = next_state 
                sum_reward += reward # TODO: might not be correct. 
        
        return sum_reward / self.trial_count 
    
    def get_random_action( self , state ) : 
        possible_actions = self.get_the_possible_actions( state ) 
        # Sample an action from this list, each action has equal probability. 
        action = random.choice( possible_actions ) 
        return action 

    def save_q_table_pickle(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f) 

    def load_q_table_pickle(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f) 

    def test_demonstration( self , just_lower = False ) : 
        
        sum_reward = 0 
        # Loop each trial count. 
        for _ in range( 1 ) : 
            game = game_module.Game() 
            if just_lower == True : 
                state = game.resetHalf() 
            else : 
                state  = game.reset() 
            is_terminal = False 
            # Loop for each step of the episode. 
            while is_terminal == False : 
                print( f"state={state}.")
                action = self.get_the_best_action( state ) 
                print( f"action={action}.")
                next_state , reward , is_terminal = game.step( action ) 
                print( f"reward={reward}.")
                state = next_state 
                sum_reward += reward # TODO: might not be correct. 
        print(f"The score of the agent: {sum_reward}. ")

    def create_demonstration( self , the_name = None , iteration = 1 , just_lower = False ) : 
        if the_name != None : 
            self.load_q_table_pickle(the_name) 
        for i in range( iteration ) : 
            print( f"Iteration: {i}. _____________________________________________________________ ") 
            self.test_demonstration( just_lower ) 

    def evaluate( self , eval_trial_count = 1000 , just_lower = False ) : 
        sum_reward = 0 
        # Loop each trial count. 
        for _ in range( eval_trial_count ) : 
            game = game_module.Game() 
            if just_lower == True : 
                state = game.resetHalf() 
            else : 
                state  = game.reset() 
            is_terminal = False 
            # Loop for each step of the episode. 
            while is_terminal == False : 
                action = self.get_the_best_action( state ) 
                next_state , reward , is_terminal = game.step( action ) 
                state = next_state 
                sum_reward += reward # TODO: might not be correct. 

        return sum_reward / eval_trial_count 

def main() : 

    """# Train just lower. 
    print("________________________ The Just Lower ________________") 
    agent = TQ_agent( 0.3 , 0.3 , 100 , 10000 , 10000 ) 
    agent.train_test( just_lower= True ) 
    agent.create_demonstration( None , 1 , just_lower= True) 
    evaluation_average_reward = agent.evaluate( eval_trial_count = 1000000 , just_lower= True ) 
    print(f"1. The just lower, evaluation average reward = {evaluation_average_reward}. ") 
    #agent.save_q_table_pickle( "just_lower_2" ) """

    # Train not just lower. 
    print("________________________ The Not Just Lower ________________") 
    agent = TQ_agent( 0.3 , 0.3 , 100 , 10000 , 10000 ) 
    agent.train_test( just_lower= False ) 
    agent.create_demonstration( None , 1 , just_lower= False) 
    evaluation_average_reward = agent.evaluate( eval_trial_count = 1000000 , just_lower= False ) 
    print(f"1. The not just lower, evaluation average reward = {evaluation_average_reward}. ") 
    #agent.save_q_table_pickle( "not_just_lower_2" ) 

if __name__ == "__main__" : 
    main() 

    