
from chess import ChessModel
import random
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import cProfile
import pstats
import os
import time

CUTOFF_DEPTH = 2
MAX_THREADS = os.cpu_count() - 1

EMPTY = "."
WHITE_KING = "K"
WHITE_QUEEN = "Q"
WHITE_ROOK = "R"
WHITE_BISHOP = "B"
WHITE_KNIGHT = "N"
WHITE_PAWN = "P"
BLACK_KING = "k"
BLACK_QUEEN = "q"
BLACK_ROOK = "r"
BLACK_BISHOP = "b"
BLACK_KNIGHT = "n"
BLACK_PAWN = "p"

class Search:

    def __init__(self):
        self.player = None
        self.board = None
        self.transposition_table = {}
        self.model = ChessModel()
        self.max_time = 5.0
    
    def MINIMAX(self,initial_state, env,player, zobrist_hash):
        self.board = initial_state
        self.player = player
        self.model = env.model
        self.zobrist = env.zobrist
        
        if zobrist_hash in self.transposition_table:
            cached_action = self.transposition_table[zobrist_hash][3]
            return (*cached_action[0], *cached_action[1])
        
        start_time = time.time()
        best_action = None
        for depth in range(1,CUTOFF_DEPTH+1):
            if time.time()- start_time > self.max_time:
                break
            best_value = float('-inf')
            current_best_action = None
            alpha = float('-inf')
            beta = float('inf')
            
            moves = self.order_moves(self.model.actions(initial_state, player), initial_state, player)
            if depth ==1:
            # Use multi-threading only at first depth as to keep optimization
                with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                    futures = {
                        
                        executor.submit(self.MIN, new_board, 1, alpha, beta,new_hash): action
                        for action in moves
                        for new_board,new_hash in [self.model.results(initial_state, action, player, zobrist_hash)]
                    }
                    for future in futures:
                        value = future.result()
                        if value > best_value:
                            best_value = value
                            current_best_action = futures[future]
                            alpha = max(alpha, best_value)
            else:
                for action in moves:
                    next_state,new_hash = self.model.results(initial_state, action, player,zobrist_hash)
                    value = self.MIN(next_state, 1, alpha, beta,new_hash)
                    if value > best_value:
                        best_value = value
                        current_best_action = action
                        alpha = max(alpha, best_value)
            if current_best_action:
                best_action = current_best_action
                self.transposition_table[zobrist_hash] = (best_value,alpha,beta,best_action)
                
        if best_action:
            return (*best_action[0], *best_action[1])
        return None
        
    def MAX(self,current_state, depth, alpha, beta, zobrist_hash):
        if zobrist_hash in self.transposition_table:
            cached_value, cached_alpha, cached_beta, _ = self.transposition_table[zobrist_hash]
            if cached_alpha >= beta:
                return cached_alpha
            if cached_beta <= alpha:
                return cached_beta
        
        original_history = self.model.get_history()
        
        if self.model.is_checkmate(current_state,self.player):
            return float("inf")
        if self.model.is_checkmate(current_state, other_player(self.player)):
            return float('-inf')
        if self.model.is_stalemate(current_state, self.player):
            return 0
        
        if depth >= CUTOFF_DEPTH :
            result = self.quiescence(current_state, depth, alpha, beta, self.player,zobrist_hash)
            self.model.set_history(original_history)
            return result
        
        best_value = float('-inf')
        for action in self.order_moves(self.model.actions(current_state, self.player), current_state, self.player):
            next_state,new_hash = self.model.results(current_state, action, self.player,zobrist_hash)
            value = self.MIN(next_state, depth + 1, alpha, beta,new_hash)
            if value > best_value:
                best_value = value
                best_action = action
                alpha = max(alpha, best_value)

            if beta <= alpha:
                break
            
        self.model.set_history(original_history)
        self.transposition_table[zobrist_hash] = (best_value, alpha, beta, best_action)
        return best_value

    def MIN(self,current_state, depth, alpha, beta, zobrist_hash):
        if  zobrist_hash in self.transposition_table:
            cached_value, cached_alpha, cached_beta, _ = self.transposition_table[zobrist_hash]
            if cached_alpha >= beta:
                return cached_alpha
            if cached_beta <= alpha:
                return cached_beta
        
        player = other_player(self.player)
        original_history = self.model.get_history()
        
        if self.model.is_checkmate(current_state,player):
            return float("inf")
        
        if self.model.is_checkmate(current_state, self.player):
            return float('-inf')
        if self.model.is_stalemate(current_state, player):
            return 0
        
        if depth >= CUTOFF_DEPTH:
            result= self.quiescence(current_state, depth, alpha, beta, player, zobrist_hash)
            self.model.set_history(original_history)
            return result
        
        best_value = float('inf')
        for action in self.order_moves(self.model.actions(current_state, player), current_state, player):
            next_state,new_hash = self.model.results(current_state, action, player, zobrist_hash)
            value = self.MAX(next_state, depth + 1, alpha, beta,new_hash)
            if value < best_value:
                best_value = value
                best_action = action
                beta = min(beta, best_value)
            if beta <= alpha:
                break
        self.model.set_history(original_history)
        self.transposition_table[zobrist_hash] = (best_value, alpha, beta, best_action)
        return best_value
    
    def quiescence(self, current_state, depth, alpha, beta, player,zobrist_hash):
        # MAX_DEPTH = 4
        # if depth >= MAX_DEPTH:
        #     return self.model.evaluate(current_state, player)
        
        stand_pat = self.model.evaluate(current_state, player)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        for action in self.model.actions(current_state, player):
            if not self.model.is_tactical_action(action, current_state,player):
                continue
            next_state,new_hash = self.model.results(current_state, action, player,zobrist_hash)
            value = -self.quiescence(next_state, depth+1, -beta, -alpha, other_player(player),new_hash)
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




class AgentAlphaBeta:

    def __init__(self):
        self.search = Search()
        self.player = 0
        self.model = ChessModel()
    
    def reset(self):
        return
    
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
        zobrist_hash = observation["zobrist_hash"]
        action = self.search.MINIMAX(board,env, player,zobrist_hash)
        action = self.flatten_action(action)
        print("agent action:",action)
        return action    

