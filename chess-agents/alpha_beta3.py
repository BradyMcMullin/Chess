
from chess import ChessModel
import random
from functools import lru_cache

CUTOFF_DEPTH = 2
class Search:

    def __init__(self):
        self.player = None
        self.board = None
        self.action_table = {}
        self.value_table = {}
        self.model = ChessModel()
        return
    
    def MINIMAX(self,initial_state, model,player):
        self.board = initial_state
        self.player = player
        self.model = model
        state_key = state_to_key(initial_state)

        if state_key in self.action_table:
            return self.action_table[state_key]

        alpha = float('-inf')
        beta = float('inf')
        best_value = float('-inf')
        best_actions = []

        for action in self.order_moves(self.model.actions(initial_state,player),initial_state,player):
            next_state,_ = self.model.results(initial_state,action,player)
            if self.model.is_repetition(next_state):
                continue
            value = self.MIN(next_state, 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_actions = [action]
                alpha = max(alpha,best_value)
            elif value == best_value:
                best_actions.append(action)
            if best_value >= beta:
                break
            
        if not best_actions:
            print("No valid actions available")
            return None
        
        chosen_action = random.choice(best_actions)
        self.action_table[state_key] = chosen_action
        return (*chosen_action[0], *chosen_action[1])
        
    def MAX(self,current_state, depth, alpha, beta):
        state_key = state_to_key(current_state)
        if (state_key,depth) in self.value_table:
            return self.value_table[(state_key,depth)]
        
        if self.model.is_checkmate(current_state,self.player):
            return float("inf")
        if self.model.is_checkmate(current_state, other_player(self.player)):
            return float('-inf')
        if self.model.is_stalemate(current_state, self.player):
            return 0
        
        if depth >= CUTOFF_DEPTH :
            return self.quiescence(current_state, 0, alpha, beta, self.player)
        
        best_value = float('-inf')
        for action in self.order_moves(self.model.actions(current_state, self.player), current_state, self.player):
            next_state,_ = self.model.results(current_state, action,self.player)
            value = self.MIN(next_state, depth+1, alpha, beta)
            best_value = max(best_value, value)
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break
            
        self.value_table[(state_key,depth)] = best_value
        return best_value

    def MIN(self,current_state, depth, alpha, beta):
        state_key = state_to_key(current_state)
        if (state_key,depth) in self.value_table:
            return self.value_table[(state_key,depth)]
        
        player = other_player(self.player)
        
        if self.model.is_checkmate(current_state,player):
            return float("inf")
        if self.model.is_checkmate(current_state, self.player):
            return float('-inf')
        if self.model.is_stalemate(current_state, player):
            return 0
        
        if depth >= CUTOFF_DEPTH:
            return self.quiescence(current_state, 0, alpha, beta, player)
        
        best_value = float('inf')
        for action in self.order_moves(self.model.actions(current_state, player), current_state, player):
            next_state,_ = self.model.results(current_state, action,player)
            value = self.MAX(next_state, depth+1, alpha, beta)
            best_value = min(best_value, value)
            beta = min(beta, best_value)
            if beta <= alpha:
                break
        
        self.value_table[(state_key,depth)] = best_value
        return best_value
    
    def quiescence(self, current_state, depth, alpha, beta, player):
        MAX_DEPTH = 4
        if depth>= MAX_DEPTH:
            return self.model.evaluate(current_state,player)
        
        if self.model.is_checkmate(current_state, player):
            return float('-inf') 
        if self.model.is_checkmate(current_state, other_player(player)):
            return float('inf') 
        if self.model.is_stalemate(current_state, player):
            return 0 
        
        stand_pat = self.model.evaluate(current_state, player)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        for action in self.model.actions(current_state, player):
            if not self.model.is_tactical_action(action, current_state,player):
                continue
            next_state,_ = self.model.results(current_state, action, player)
            value = -self.quiescence(next_state, depth+1, -beta, -alpha, other_player(player))
            if value >= beta:
                return beta
            if value > alpha:
                alpha = value
        return alpha
    
    def order_moves(self, actions, current_state, player):
        def move_priority(action):
            start_pos, end_pos = action
            target = current_state[end_pos[0]][end_pos[1]]
            attacker = current_state[start_pos[0]][start_pos[1]]
            
            if target != ".":
                return 10 + self.model.piece_value(target)- self.model.piece_value(attacker)
            if self.model.is_in_check_after_move(current_state,start_pos,end_pos,player):
                return 8
            if self.model.is_promotion(current_state,start_pos,end_pos,player):
                return 7
            if (end_pos in [(3, 3), (3, 4), (4, 3), (4, 4)]):
                return 5
            return 0
        return sorted(actions, key=move_priority, reverse=True)
    

def state_to_key(state):
    immutable_state = tuple(tuple(row) for row in state)
    return state_to_key_cache(immutable_state)

@lru_cache(maxsize=10000)
def state_to_key_cache(immutable_state):
    return immutable_state
    
def other_player(player):
    return "White" if player == "Black" else "Black"

class AgentAlphaBetav3:

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

