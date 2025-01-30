import chess 
import random

class AgentRandom:
    
    def __init__(self):
        pass
    def reset(self):
        pass
    
    def agent_function(self,observation, env,agent):
        board = observation["board"]
        valid_actions = env.model.actions(board,agent)
            
        if valid_actions:
            test_action = None
            start_pos, end_pos = random.choice(valid_actions)
            new_board,_ = env.model.results(board, (start_pos,end_pos), agent,None, True)
            while env.model.is_repetition(new_board):
                start_pos, end_pos = random.choice(valid_actions)
                new_board,_ = env.model.results(board, (start_pos,end_pos), agent,None, True)
            action = (*start_pos, *end_pos)  # Flatten (start_pos, end_pos) into action format (x1, y1, x2, y2)
        else:
            action = None  # No valid moves left

            # Step with the chosen action
        return action