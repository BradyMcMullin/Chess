import chess as rps
class AgentHuman:
    def __init__(self):
        self.env = rps.env(render_mode = "human")
        self.env.reset(seed=42) 
    
    def reset(self):
        self.env.reset(seed=42)
        
    def agent_function(self,observation,info,agent):
        board = observation["board"]
        valid_actions = self.env.model.actions(board,agent)
        if valid_actions:
            test_action = None
            while test_action not in valid_actions:
                x1, y1, x2, y2 = map(int, input("Make a move with format x1,y1,x2,y2: ").split(','))
                test_action = ((x1,y1),(x2,y2))
                action = (x1,y1,x2,y2)  # Flatten (start_pos, end_pos) into action format (x1, y1, x2, y2)
        else:
            action = None  # No valid moves left
            "wack"

        # Step with the chosen action
        print(action)
        return action