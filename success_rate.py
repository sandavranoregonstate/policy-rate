import re
import numpy as np 

performance_path = [ 

    [ "performance/1/1/50/0.txt" , 50 , "Policy Rate: 50, State Noise: 0. " ] , 
    [ "performance/1/1/100/0.txt" , 100 , "Policy Rate: 100, State Noise: 0. " ] , 

    [ "robust/1/1/50/0-1.txt" , 50 , "Policy Rate: 50, State Noise: 0.1. " ] , 
    [ "robust/1/1/50/0-2.txt" , 50 , "Policy Rate: 50, State Noise: 0.2. " ] , 
    [ "robust/1/1/50/0-3.txt" , 50 , "Policy Rate: 50, State Noise: 0.3. " ] , 
    
    [ "robust/1/1/100/0-1.txt" , 100 , "Policy Rate: 100, State Noise: 0.1. " ] , 
    [ "robust/1/1/100/0-2.txt" , 100 , "Policy Rate: 100, State Noise: 0.2. " ] , 
    [ "robust/1/1/100/0-3.txt" , 100 , "Policy Rate: 100, State Noise: 0.3. " ] 

] 

success_rate = [] 
MAR_list = [] 

for csv_path , policy_rate , policy_name in performance_path : 

    # Open and read the file
    with open(csv_path, 'r') as file:
        # Lists to store the extracted data
        episode_length_list = []
        average_reward_list = []
        sample_output = file.read() 
        # Regex to find episode lengths and average rewards 
        episode_length_pattern = r"Episode length = (\d+)"
        average_reward_pattern = r"Average reward is ([\d\.]+)"
        # Finding all matches in the file content
        episode_length_list = re.findall(episode_length_pattern, sample_output)
        average_reward_list = re.findall(average_reward_pattern, sample_output)
        # Convert found strings to integers for episode lengths and floats for rewards
        episode_length_list = [int(length) for length in episode_length_list]
        average_reward_list = [float(reward.rstrip('.')) for reward in average_reward_list] 
        # Find the averages. 
        trial_count = 10 
        episode_length_list = episode_length_list[:trial_count] 
        average_reward_list = average_reward_list[:trial_count] 
        # Calculate the number of elements greater than the threshold 
        threshold = 6 * policy_rate 
        count_greater = np.sum(np.array(episode_length_list) >= threshold)
        # Calculate the percentage of numbers greater than the threshold
        ratio_greater = (count_greater / len( episode_length_list ) ) 
        success_rate.append( ( ratio_greater , policy_name ) ) 
        MAR_list.append( ( np.mean( average_reward_list ) , policy_name ) ) 

print( success_rate ) 
print( MAR_list ) 

