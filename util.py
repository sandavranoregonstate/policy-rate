def create_graph( x_len , y_len , csv_list , type , colors ) : 

    print(type)

    # install the dependencies. 
    import pandas as pd 
    import matplotlib.pyplot as plt 
    plt.style.use('ggplot') 

    if type == 1 : 
        # Create the 1st plot. (A) 
        # Sim Time vs. Test Average Reward. 
        plt.figure(figsize=(x_len,y_len)) 
        i = 0 
        for csv_file_path , policy_rate in csv_list : 
            df = pd.read_csv(csv_file_path) 
            df.replace( {"Test/Return" : ""} , pd.NA , inplace= True)
            df = df.dropna(subset=['Test/Return']) 
            # Create the average return. 
            df['Test/Return'] /= ( 6 * policy_rate ) 
            # Create the sim time. 
            df["Time/Total Timesteps"] /= policy_rate
            plt.scatter(df['Time/Total Timesteps'], df['Test/Return'] , color = colors[ i ])
            plt.plot(df['Time/Total Timesteps'], df['Test/Return'], label=f"{policy_rate}Hz" , color = colors[ i ]) 
            i += 1 
        plt.title('Sim Time vs. Test Average Reward. ') 
        plt.xlabel('Sim Time. (s)')
        plt.ylabel('Test Average Reward. ') 
        plt.legend() 
        plt.show() 
    elif type == 2 : 
        # Create the 1st plot. (B) 
        # Sim Time vs. Test Episode Length. 
        plt.figure(figsize=(x_len,y_len)) 
        i = 0 
        for csv_file_path , policy_rate in csv_list : 
            df = pd.read_csv(csv_file_path) 
            df.replace( {"Test/Episode Length" : ""} , pd.NA , inplace= True)
            df = df.dropna(subset=['Test/Episode Length']) 
            # Create the episode length in seconds. 
            df["Test/Episode Length"] /= policy_rate 
            # Create the sim time. 
            df["Time/Total Timesteps"] /= policy_rate
            plt.scatter(df['Time/Total Timesteps'], df['Test/Episode Length'], label=f"{policy_rate}Hz"  , color = colors[ i ])
            plt.plot(df['Time/Total Timesteps'], df['Test/Episode Length'] , color = colors[ i ])
            i += 1 
        plt.title('Sim Time vs. Test Episode Length. ') 
        plt.xlabel('Sim Time. (s) ')
        plt.ylabel('Test Episode Length. (s) ') 
        plt.legend() 
        plt.show() 
    elif type == 3 : 
        # Create the 2nd plot. (A)
        # Relative wall/process time vs. Test Average Reward. 
        # TODO: Not sure if this is working. 
        plt.figure(figsize=(x_len,y_len)) 
        i = 0 
        for csv_file_path , policy_rate in csv_list : 
            df = pd.read_csv(csv_file_path) 
            df['Real Time, Relative'] = df['Time/Timesteps per Iteration'] / df["Time/Timesteps per Second (FULL)"] 
            df['Real Time, Cumulative'] = df['Real Time, Relative'].cumsum() 
            df.replace( {"Test/Return" : ""} , pd.NA , inplace= True)
            df = df.dropna(subset=['Test/Return']) 
            # Create the average return. 
            df['Test/Return'] /= ( 6 * policy_rate ) 
            plt.scatter(df['Real Time, Cumulative'], df['Test/Return'], label=f"{policy_rate}Hz" , color = colors[ i ])
            plt.plot(df['Real Time, Cumulative'], df['Test/Return'] , color = colors[ i ]) 
            i += 1 
        plt.title('Real Time vs. Test Average Reward. ') 
        plt.xlabel('Real Time. (s)')
        plt.ylabel('Test Average Reward. ') 
        plt.legend() 
        plt.show() 
    elif type == 4 : 
        # Create the 2nd plot. (B)
        # Relative wall/process time vs. Test Episode Length. 
        # TODO: Not sure if this is working. 
        plt.figure(figsize=(x_len,y_len)) 
        i = 0 
        for csv_file_path , policy_rate in csv_list : 
            df = pd.read_csv(csv_file_path) 
            df['Real Time, Relative'] = df['Time/Timesteps per Iteration'] / df["Time/Timesteps per Second (FULL)"] 
            df['Real Time, Cumulative'] = df['Real Time, Relative'].cumsum()     
            df.replace( {"Test/Episode Length" : ""} , pd.NA , inplace= True)
            df = df.dropna(subset=['Test/Episode Length']) 
            # Create the episode length in seconds. 
            df["Test/Episode Length"] /= policy_rate
            plt.scatter(df['Real Time, Cumulative'], df['Test/Episode Length'] , label=f"{policy_rate}Hz" , color = colors[ i ])
            plt.plot(df['Real Time, Cumulative'], df['Test/Episode Length']  , color = colors[ i ]) 
            i += 1 
        plt.title('Real Time vs. Test Episode Length. ') 
        plt.xlabel('Real Time. (s)')
        plt.ylabel('Test Episode Length. (s)') 
        plt.legend() 
        plt.show() 

def create_eval( h , v , per_rob , main_dict , type , colors ) : 

    import os 
    import re 
    import numpy as np 
    # install the dependencies. 
    import pandas as pd 
    import matplotlib.pyplot as plt 
    plt.style.use('ggplot') 

    trial_count = 100 
    success_rate_dict = {} 
    MAR_dict = {} 
    for csv_path , policy_rate , the_noise in per_rob : 
        # Open and read the file 
        if os.path.exists(csv_path):
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
                try : 
                    episode_length_list = episode_length_list[:trial_count] 
                    average_reward_list = average_reward_list[:trial_count] 
                except : 
                    episode_length_list = episode_length_list[:trial_count] 
                    average_reward_list = average_reward_list[:trial_count] 
                # Calculate the number of elements greater than the threshold 
                threshold = 6 * policy_rate 
                count_greater = np.sum(np.array(episode_length_list) >= threshold)
                # Calculate the percentage of numbers greater than the threshold
                ratio_greater = (count_greater / len( episode_length_list ) ) * 100 
                if policy_rate not in success_rate_dict : 
                    success_rate_dict[ policy_rate ] = { "x" : [ the_noise ], "y" : [ ratio_greater ] } 
                else : 
                    success_rate_dict[ policy_rate ][ "x" ].append( the_noise ) 
                    success_rate_dict[ policy_rate ][ "y" ].append( ratio_greater ) 

                if policy_rate not in MAR_dict : 
                    MAR_dict[ policy_rate ] = { "x" : [ the_noise ] , "y" : [ np.mean( average_reward_list ) ] } 
                else : 
                    MAR_dict[ policy_rate ]["x"].append( the_noise ) 
                    MAR_dict[ policy_rate ]["y"].append( np.mean( average_reward_list ) ) 

    if type != 0 : 
        main_dict[ type ] = success_rate_dict.copy() 

    plt.figure(figsize=(h,v)) 
    i = 0 
    for policy_name in success_rate_dict : 
        pop_count = 0 
        for idx in range( len( success_rate_dict[ policy_name ]["y"] ) ) : 
            if np.isnan( success_rate_dict[ policy_name ]["y"][ idx-pop_count ] ) : 
                success_rate_dict[ policy_name ]["x"].pop(idx - pop_count ) 
                success_rate_dict[ policy_name ]["y"].pop(idx -pop_count) 
                pop_count += 1 
        plt.plot(success_rate_dict[ policy_name ]["x"], success_rate_dict[ policy_name ]["y"] , color = colors[ i ]) 
        plt.scatter(success_rate_dict[ policy_name ]["x"], success_rate_dict[ policy_name ]["y"], label=f"{policy_name}Hz" , color = colors[ i ] ) 
        i += 1 
    plt.title('State Noise vs. Success Rate. ') 
    plt.xlabel('State Noise. ') 
    plt.ylabel('Success Rate. (%)') 
    plt.legend() 
    plt.show() 

    plt.figure(figsize=(h,v)) 
    i = 0 
    for policy_name in MAR_dict : 
        pop_count = 0 
        for idx in range( len( MAR_dict[ policy_name ]["y"] ) ) : 
            if np.isnan( MAR_dict[ policy_name ]["y"][ idx-pop_count ] ) : 
                MAR_dict[ policy_name ]["x"].pop(idx - pop_count ) 
                MAR_dict[ policy_name ]["y"].pop(idx -pop_count) 
                pop_count += 1 
        plt.plot(MAR_dict[ policy_name ]["x"], MAR_dict[ policy_name ]["y"] , color = colors[ i ]) 
        plt.scatter(MAR_dict[ policy_name ]["x"], MAR_dict[ policy_name ]["y"], label=f"{policy_name}Hz" , color = colors[ i ]) 
        i += 1 
    plt.title('State Noise vs. Average Reward. ') 
    plt.xlabel('State Noise. ') 
    plt.ylabel('Average Reward. ') 
    plt.legend() 
    plt.show() 

    

# create the list. 
path_policy_rate_list = [
    [ 'graph/1/1/50.csv' , 50 ] , 
    [ "graph/1/1/100.csv" , 100 ] , 
    [ "graph/1/1/200.csv" , 200 ] , 
    [ "graph/1/1/400.csv" , 400 ] , 
    [ "graph/1/1/800.csv" , 800 ] 
] 

def main() : 

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:brown', 'tab:pink']  # Define a list of colors 
    create_graph( 6 , 4 , path_policy_rate_list.copy() , 1 , colors ) 
    create_graph( 6 , 4 , path_policy_rate_list.copy() , 2 , colors ) 
    create_graph( 6 , 4 , path_policy_rate_list.copy() , 3 , colors ) 
    create_graph( 6 , 4 , path_policy_rate_list.copy() , 4 , colors ) 


if __name__ == "__main__" : 
    main() 


