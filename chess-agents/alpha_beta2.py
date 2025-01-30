
from chess import ChessModel
import random
CUTOFF_DEPTH = 2
class Search:

    def __init__(self):
        self.player = None
        self.board = None
        self.table = {}
        self.model = ChessModel()
        return
    
    def MINIMAX(self,initial_state,model,player):
        self.board = initial_state
        self.player = player
        self.model = model
        state_key = self.state_to_key(self.board)

        if state_key in self.table:
            return self.table[state_key]

        alpha = float('-inf')
        beta = float('inf')
        best_value = float('-inf')
        current_state = initial_state
        best_actions = None
        #self.state = current_state
        #print("possible actions:",self.model.actions(current_state,player))
        for action in self.model.actions(current_state,player):
            next_state,_ = self.model.results(current_state,action,player)
            if self.model.is_repetition(next_state):
                continue
            value = self.MIN(next_state, 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_actions = [action]
                if best_value > alpha:
                    alpha = best_value
                if best_value >= beta:
                    break
            elif value == best_value:
                best_actions.append(action)
        if not best_actions:
            print("No valid actions available")
            return None
        chosen_action = random.choice(best_actions)
        self.table[state_key] = chosen_action
        flattened_tuple = (*chosen_action[0], *chosen_action[1])
        return flattened_tuple
        
    def MAX(self,current_state, depth, alpha, beta):
        if depth >= CUTOFF_DEPTH or self.model.is_checkmate(current_state, self.player):
            return self.model.evaluate(current_state, self.player)
        
        best_value = float('-inf')

        for action in self.model.actions(current_state, self.player):
            next_state,_ = self.model.results(current_state, action,self.player)
            value = self.MIN(next_state, depth+1, alpha, beta)

            if value > best_value:
                best_value = value
                if best_value > beta:
                    break
                if best_value > alpha:
                    alpha = best_value

            if beta <= alpha:
                break
        return best_value

    def MIN(self,current_state, depth, alpha, beta):
        player = other_player(self.player)

        if depth >= CUTOFF_DEPTH or self.model.is_checkmate(current_state, player):
            return self.model.evaluate(current_state,player)
        
        best_value = float('inf')

        for action in self.model.actions(current_state,player):
            next_state, _ = self.model.results(current_state, action,player)
            value = self.MAX(next_state, depth+1, alpha, beta)

            if value < best_value:
                best_value = value
                if best_value < alpha:
                    break
                if best_value < beta:
                    beta = best_value

            if beta <= alpha:
                break
        return best_value
    
    def state_to_key(self,state):
        return str(state)
    
def other_player(player):
    return "White" if player == "Black" else "Black"
class AgentAlphaBetav2:

    def __init__(self):
        self.search = Search()
        self.player = 0
        self.model = ChessModel()
    
    def reset(self):
        self.search = Search()
    
    def is_flattened_action(self,action):
     # Check if action is a tuple or list of length 4 and contains integers
        return isinstance(action, (tuple, list)) and len(action) == 4 and all(isinstance(i, int) for i in action)
        
    def flatten_action(self,action):
    # If the action is already flattened, return it
        if self.is_flattened_action(action):
            return action
        # If action is a tuple of tuples ((x1, y1), (x2, y2)), flatten it
        if isinstance(action, (tuple, list)) and len(action) == 2:
            start_pos, end_pos = action
            if isinstance(start_pos, (tuple, list)) and isinstance(end_pos, (tuple, list)):
                return (*start_pos, *end_pos)
        raise ValueError(f"Cannot flatten action: {action}")
    
    def agent_function(self,observation,env,player):
        board = observation["board"]
        model = env.model
        action = self.search.MINIMAX(board,model, player)
        action = self.flatten_action(action)
        print("agent action:",action)
        return action       

