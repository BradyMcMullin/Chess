import functools 
import copy
import gymnasium
import numpy as np
import random
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

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
MOVES=["WHITE","BLACK"]
NUM_ITERS = 500 #update later to make it so it only ends if stalemate! (50 concecutive moves of no captures and no pawn moves)
REWARD_MAP = {
    "win":1000,
    "loss":-1000,
    "draw":0,
    "pawn":1,
    "knight": 3,
    "bishop": 3, 
    "rook": 5, 
    "queen": 9, 
    "king":10,
    "promotion_self":9, "promotion_opponent":-9,
    "check_opponent": 51, "check_self":-50,
    "illegal_move": -50 #just in case :)
}

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode = internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    
    metadata = {"render_modes": ["human"], "name":"chess_v1"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["White","Black"]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}
        self.model = ChessModel()
        
        self._action_spaces = {agent: MultiDiscrete([8,8,8,8]) for agent in self.possible_agents} # using multidiscrete for simplicity, set up is (x1,y1) -> (x2,y2)/start pos, end pos
        self._observation_spaces = {agent: MultiDiscrete([12 for _ in range(8*8)]) for agent in self.possible_agents} # 12 possible actions for each 8*8 position on the board, change the 12 if need more or less actions
        
        self.board = self.model.initialize_board()
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent:False for agent in self.agents}
        self.render_mode = render_mode
        self.state_history = {}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.zobrist = Zobrist()
        self.zobrist_hash = self.zobrist.calculate_initial_hash(self.board)
          
    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn("You are calling render method without specifying any render mode.")
            return
        
         # Display the current state of the board
        print("Current State of the Chessboard:")
        self.model.render_board(self.board)

        # Check for win/loss/draw conditions
        if all(self.terminations.values()):
            if all(r == 0 for r in self.rewards.values()):
                print("Game ends in a draw.")
            elif self.rewards[self.agents[0]] > self.rewards[self.agents[1]]:
                print("White Wins!")
            else:
                print("Black Wins!")

        # Display rewards and move count
        print("Agent reward:", self.rewards)
        print("Move count:", self.num_moves)
        
    
    def observe(self,agent):
        observation = {
            "board":self.board,
            "zobrist_hash":self.zobrist_hash
        }
        return observation
    
    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.model = ChessModel()
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.board = self.model.initialize_board()
        self.state_history = {}
        self.zobrist = Zobrist()
        self.zobrist_hash = self.zobrist.calculate_initial_hash(self.board)

    
    def step(self, action):
        if not hasattr(self, 'zobrist_hash'):
            self.zobrist_hash = self.zobrist.calculate_initial_hash(self.board)
        action = self.flatten_action(action)
        if self.agent_selection not in self._cumulative_rewards:
            self._cumulative_rewards[self.agent_selection] = 0

        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        other_agent = self.get_other_agent(agent)
        
        start_pos, end_pos = (action[:2], action[2:])
        reward = 0

        if self.model.is_legal_move(self.board, start_pos, end_pos, agent):
            
            self.board,self.zobrist_hash = self.model.results(self.board, (start_pos, end_pos), agent,self.zobrist_hash)
            self.model.record_board_state(self.board)
            
            if self.model.is_repetition(self.board):
                print("Threefold repetition detected. Game ends in a draw.")
                for a in self.agents:
                    self.terminations[a]= True
                    self.rewards[a] = 0
                return
            
            reward = self.model.evaluate(self.board, agent)
            
            if self.model.is_stalemate(self.board,other_agent):
                for a in self.agents:
                    self.terminations[a] = True
                    self.rewards[a] = 0    
                return  
            elif self.model.is_checkmate(self.board,other_agent):
                print(f"Checkmate! {agent} wins.")
                for a in self.agents:
                    self.terminations[a] = True
                self.rewards[agent] = 1
                self.rewards[other_agent] = -1
                return      
        else:
            print("illegal move",agent,": ",action)
            self.terminations[agent] = True
            self.rewards[agent] = -1
            
        # Update the agent's rewards and state
        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward
        self.state[agent] = action

        #check if all agents have taken turn
        if self._agent_selector.is_last():
            self.num_moves+=1
            self.truncations = {agent: self.num_moves >= NUM_ITERS for agent in self.agents}
        
        #Select the next agent
        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
                self.render()

    
    def last(self):
        observation = self.observe(self.agent_selection)
        reward = self.rewards.get(self.agent_selection, 0)
        termination = self.terminations[self.agent_selection]
        truncation = self.truncations[self.agent_selection]
        info = {
            "legal_moves": self.legal_moves(self.agent_selection),
            "actions_history": self.actions_history,
            "zobrist":self.zobrist
        }
        return observation, reward, termination, truncation, info
        

    def _advance_turn(self):
        self.agent_selection = self._agent_selector.next()

    def get_other_agent(self, agent):
        return "Black" if agent == "White" else "White"
    
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
class ChessModel:
    def __init__(self):
        self.board = self.initialize_board()
        self.num_moves = 0
        self.possible_agents = ["White","Black"]
        self.actions_history = {agent: [] for agent in self.possible_agents}
        self.board_state_history = []
        self.zobrist = Zobrist()

    def reset(self):
        self.board = self.initialize_board()
        self.num_moves = 0
        self.actions_history = {agent: [] for agent in self.possible_agents}    
        self.board_state_history = []
        self.zobrist = Zobrist()
    
    def record_board_state(self,board):
        board_tuple = tuple(tuple(row) for row in board)
        self.board_state_history.append(board_tuple)
        
    def is_repetition(self,board):
        board_tuple = tuple(tuple(row) for row in board)
        return self.board_state_history.count(board_tuple) >= 3
    
    def initialize_board(self):
        board = [["." for _ in range(8)] for _ in range(8)]
        board[0] = [BLACK_ROOK, BLACK_KNIGHT, BLACK_BISHOP, BLACK_QUEEN, BLACK_KING, BLACK_BISHOP, BLACK_KNIGHT, BLACK_ROOK]
        board[1] = [BLACK_PAWN] * 8
        board[6] = [WHITE_PAWN] * 8
        board[7] = [WHITE_ROOK, WHITE_KNIGHT, WHITE_BISHOP, WHITE_QUEEN, WHITE_KING, WHITE_BISHOP, WHITE_KNIGHT, WHITE_ROOK]
        return board
    
    def render_board(self, board):
        for row in board:
            print(" ".join(row))
    
    def is_legal_move(self, board, start_pos, end_pos, agent):
        piece = board[start_pos[0]][start_pos[1]]
        x1, y1 = start_pos
        x2,y2 = end_pos

        color = agent

        if start_pos == end_pos:
            return False
        if board[start_pos[0]][start_pos[1]] == EMPTY:
            return False

        #check for pieces then check moves accordingly
        if piece in (BLACK_KING, WHITE_KING): #update to add castling
            #check up down left right and out of bounds
            if (x1,y1) == (x2+1,y2) or (x1,y1) == (x2-1,y2) or (x1,y1) == (x2,y2+1) or (x1,y1) == (x2,y2-1) and 0<=x2<8 and 0<=y2<8 or\
                (x1,y1) == (x2+1,y2+1) or (x1,y1) == (x2+1,y2-1) or (x1,y1) == (x2-1,y2+1) or (x1,y1) == (x2-1,y2-1):
                if board[end_pos[0]][end_pos[1]] == EMPTY:
                    return True
                elif board[end_pos[0]][end_pos[1]].islower() and color == "White":
                    return True
                elif board[end_pos[0]][end_pos[1]].isupper() and color == "Black":
                    return True 
                elif self.is_castle(board,agent, start_pos, end_pos):
                    return True
        elif piece in (BLACK_QUEEN, WHITE_QUEEN):
            if (x1==x2) or (y1 == y2) or (abs(x1-x2) == abs(y1-y2)) and 0<=x2<8 and 0<=x2<8:
                if self.is_path_blocked(board,start_pos, end_pos):
                    return False
                if board[end_pos[0]][end_pos[1]] == EMPTY:
                        return True
                elif board[end_pos[0]][end_pos[1]].islower() and color == "White":
                    return True
                elif board[end_pos[0]][end_pos[1]].isupper() and color == "Black":
                    return True
        elif piece in(BLACK_BISHOP, WHITE_BISHOP):
            if (abs(x1-x2) == abs(y1-y2)) and 0<=x2<8 and 0<=x2<8:
                if self.is_path_blocked(board,start_pos, end_pos):
                    return False
                if board[end_pos[0]][end_pos[1]] == EMPTY:
                        return True
                elif board[end_pos[0]][end_pos[1]].islower() and color == "White":
                    return True
                elif board[end_pos[0]][end_pos[1]].isupper() and color == "Black":
                    return True
        elif piece in (BLACK_KNIGHT, WHITE_KNIGHT):
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if (dx ==2 and dy ==1) or (dx == 1 and dy == 2) and 0<=x2<8 and 0<=x2<8:
                if board[end_pos[0]][end_pos[1]] == EMPTY:
                    return True
                elif board[end_pos[0]][end_pos[1]].islower() and color == "White":
                    return True
                elif board[end_pos[0]][end_pos[1]].isupper() and color == "Black":
                    return True
        elif piece in (BLACK_ROOK, WHITE_ROOK):
            if (x1==x2) or (y1 == y2) and 0<=x2<8 and 0<=x2<8:
                if self.is_castle(board,agent, start_pos, end_pos):
                    return True
                if self.is_path_blocked(board,start_pos, end_pos):
                    return False
                if board[end_pos[0]][end_pos[1]] == EMPTY:
                        return True
                elif board[end_pos[0]][end_pos[1]].islower() and color == "White":
                    return True
                elif board[end_pos[0]][end_pos[1]].isupper() and color == "Black":
                    return True
        elif piece == WHITE_PAWN:
            # One square forward
            if x1 - 1 == x2 and y1 == y2 and board[x2][y2] == EMPTY:
                return True
            # Two squares forward from starting position
            elif x1 == 6 and x1 - 2 == x2 and y1 == y2 and board[x2][y2] == EMPTY and board[x1 - 1][y1] == EMPTY:
                return True
            # Diagonal capture
            elif x1 - 1 == x2 and abs(y1 - y2) == 1 and board[x2][y2].islower():
                return True
            # En passant
            elif self.is_en_passant_capture(board,start_pos, end_pos, agent):
                return True

        elif piece == BLACK_PAWN:
            # One square forward
            if x1 + 1 == x2 and y1 == y2 and board[x2][y2] == EMPTY:
                return True
            # Two squares forward from starting position
            elif x1 == 1 and x1 + 2 == x2 and y1 == y2 and board[x2][y2] == EMPTY and board[x1 + 1][y1] == EMPTY:
                return True
            # Diagonal capture
            elif x1 + 1 == x2 and abs(y1 - y2) == 1 and board[x2][y2].isupper():
                return True
            # En passant
            elif self.is_en_passant_capture(board,start_pos, end_pos, agent):
                return True


            return False

    def make_move(self, board, start_pos, end_pos, agent):
        if not self.is_legal_move(board, start_pos, end_pos, agent) or self.is_in_check_after_move(board,start_pos, end_pos, agent):
            return "illegal"  # Indicate an illegal move for debugging

        #record move
        piece = board[start_pos[0]][start_pos[1]]
        captured = board[end_pos[0]][end_pos[1]]
        self.actions_history[agent].append((*start_pos, *end_pos))

        # Make the move permanently
        board[end_pos[0]][end_pos[1]] = piece
        board[start_pos[0]][start_pos[1]] = EMPTY

        # Handle en passant
        if self.is_en_passant_capture(board,start_pos, end_pos, agent):
            x1, y1 = end_pos
            if agent == "Black":
                board[x1 - 1][y1] = EMPTY
            else:
                board[x1 + 1][y1] = EMPTY
            captured = WHITE_PAWN if agent == "Black" else BLACK_PAWN

        # Handle castling
        elif self.is_castle(board,agent, start_pos, end_pos):
            if agent == "Black":
                if end_pos == (0, 6):  # Kingside castling
                    board[0][5] = BLACK_ROOK
                    board[0][7] = EMPTY
                elif end_pos == (0, 2):  # Queenside castling
                    board[0][3] = BLACK_ROOK
                    board[0][0] = EMPTY
            elif agent == "White":
                if end_pos == (7, 6):  # Kingside castling
                    board[7][5] = WHITE_ROOK
                    board[7][7] = EMPTY
                elif end_pos == (7, 2):  # Queenside castling
                    board[7][3] = WHITE_ROOK
                    board[7][0] = EMPTY

        # Handle promotion
        elif self.is_promotion(board,start_pos, end_pos, agent):
            board[end_pos[0]][end_pos[1]] = WHITE_QUEEN if agent == "White" else BLACK_QUEEN

        # Normalize the captured piece type for return
        if captured in [BLACK_PAWN, WHITE_PAWN]:
            captured = "pawn"
        elif captured in [BLACK_ROOK, WHITE_ROOK]:
            captured = "rook"
        elif captured in [BLACK_KNIGHT, WHITE_KNIGHT]:
            captured = "knight"
        elif captured in [BLACK_BISHOP, WHITE_BISHOP]:
            captured = "bishop"
        elif captured in [BLACK_KING, WHITE_KING]:
            captured = "king"
        elif captured in [BLACK_QUEEN, WHITE_QUEEN]:
            captured = "queen"
        else:
            captured = None

        return captured

    def actions(self, board, agent):
        possible_actions = []
        for piece_position in self.get_all_pieces(board, agent):
            piece = board[piece_position[0]][piece_position[1]]
            for move in self.get_possible_moves(board, piece, piece_position):
                if not self.is_in_check_after_move(board, piece_position, move, agent):
                    possible_actions.append((piece_position, move))
        return possible_actions
    
    def is_path_blocked(self, board, start_pos, end_pos):
        x1, y1 = start_pos
        x2,y2 = end_pos
        step_x = 0
        step_y = 0

        if x1 != x2:
            step_x = 1 if x2 > x1 else -1
        if y1 != y2:
            step_y = 1 if y2 > y1 else -1
        
        current_x, current_y = x1+step_x, y1+step_y
        while(current_x != x2 or current_y != y2):
            if board[current_x][current_y] is not EMPTY:
                return True
            current_x += step_x
            current_y += step_y
        return False

    

    def is_promotion(self, board,start_pos, end_pos, agent):
        x1,y1 = start_pos
        x2,y2 = end_pos
        if agent == "White":
            if x2 == 0 and board[x1][y1] == WHITE_PAWN:
                return True
        elif agent == "Black":
            if x2 == 7 and board[x1][y1] == BLACK_PAWN:
                return True
        return False

    def is_castle(self, board, agent, start_pos, end_pos):
        x1, y1 = start_pos
        x2, y2 = end_pos
        piece = board[start_pos[0]][start_pos[1]]

        if piece == BLACK_KING:
            #castle right side
            if x1 == 7 and y1 ==4 and x2==7 and y2==6:
                rook_start = (7,7)
                rook_end = (7,5)
                return self.is_legal_castle(board,BLACK_KING, rook_start, rook_end,agent)
            #castle left side
            elif x1 == 7 and y1 == 4 and x2 == 7 and y2 == 2: 
                rook_start = (7, 0)
                rook_end = (7, 3) 
                return self.is_legal_castle(board,BLACK_KING, rook_start, rook_end,agent)
            
        elif piece == WHITE_KING:
            #castle right side
            if x1 == 0 and y1 == 4 and x2 == 0 and y2 == 6: 
                rook_start = (0, 7)
                rook_end = (0, 5)   
                return self.is_legal_castle(board,WHITE_KING, rook_start, rook_end,agent)
            #castle left side
            elif x1 == 0 and y1 == 4 and x2 == 0 and y2 == 2: 
                rook_start = (0, 0)
                rook_end = (0, 3)
                return self.is_legal_castle(board,WHITE_KING, rook_start, rook_end,agent)
        return False
    
    def is_legal_castle(self, board, king, rook_start, rook_end, agent):
        king_pos = (7, 4) if king == BLACK_KING else (0, 4)

        for move in self.actions_history[agent]:
            x1,y1 = move[:2]
            if (x1,y1) == king_pos or (x1,y1) == rook_start:
                return False
            
        if board[rook_start[0]][rook_start[1]] not in [BLACK_ROOK, WHITE_ROOK]:
            return False
        
        x1, y1 = rook_start
        x2, y2 = rook_end
        if self.is_path_blocked(board,(x1,y1),(x2,y2)):
            return False

        for move in [(x1, 4), (x1, 5), (x1, 6)] if y2 == 6 else [(x1, 4), (x1, 3), (x1, 2)]:
            if self.is_in_check_after_move(board,king_pos,move, agent):
                return False
            
        return True
    
    def is_en_passant_capture(self,board, start_pos, end_pos, agent):
        x1, y1 = start_pos
        x2, y2 = end_pos
        piece = board[start_pos[0]][start_pos[1]]
        opponent = "White" if piece.islower() else "Black"

        if self.actions_history[opponent]:
            last_move = self.actions_history[opponent][-1]
            startx, starty, endx, endy = last_move[:]
            last_start=(startx,starty)
            last_end = (endx,endy)

            if abs(last_start[0]-last_end[0]) == 2 and abs(last_start[1] - last_end[1]) == 0:
                if last_end[0] == x1 and abs(last_end[1] - y1) == 1:
                    if (piece == BLACK_PAWN and x2 == x1 - 1 and y2 == last_end[1]) or (piece == WHITE_PAWN and x2 == x1 + 1 and y2 == last_end[1]):
                        return True
        return False

    def is_in_check_after_move(self, board, start_pos, end_pos, agent):
        piece = board[start_pos[0]][start_pos[1]]
        captured_piece = board[end_pos[0]][end_pos[1]]
        en_passant_captured_piece = None
        en_passant_position = None

        # Temporarily make the move
        board[end_pos[0]][end_pos[1]] = piece
        board[start_pos[0]][start_pos[1]] = EMPTY

        # Handle en passant (if applicable, store the captured pawn and position)
        if piece == WHITE_PAWN and start_pos[0] == 3 and end_pos[0] == 2 and abs(start_pos[1] - end_pos[1]) == 1:
            en_passant_position = (3, end_pos[1])
            en_passant_captured_piece = board[3][end_pos[1]]
            board[3][end_pos[1]] = EMPTY
        elif piece == BLACK_PAWN and start_pos[0] == 4 and end_pos[0] == 5 and abs(start_pos[1] - end_pos[1]) == 1:
            en_passant_position = (4, end_pos[1])
            en_passant_captured_piece = board[4][end_pos[1]]
            board[4][end_pos[1]] = EMPTY

        # Check if the move puts the agent in check
        in_check = self.is_in_check(board,agent)

        # Revert the move by restoring original values
        board[start_pos[0]][start_pos[1]] = piece
        board[end_pos[0]][end_pos[1]] = captured_piece

        # Restore en passant captured piece if needed
        if en_passant_captured_piece is not None:
            board[en_passant_position[0]][en_passant_position[1]] = en_passant_captured_piece

        return in_check

    def is_checkmate(self, board, agent):
        if not self.is_in_check(board, agent):
            return False
        for start_pos in self.get_all_pieces(board, agent):
            for move in self.get_possible_moves(board, board[start_pos[0]][start_pos[1]], start_pos):
                if self.is_legal_move(board, start_pos, move, agent):
                    if not self.is_in_check_after_move(board, start_pos, move, agent):
                        return False

        return True
    
    def is_in_check(self, board, agent):
        is_white = agent.lower()=="white"
        if is_white:
            king = WHITE_KING 
        else:
            king = BLACK_KING
        king_pos = self.find_king(board,king)
        if king_pos is None:
            return False
        
        opponent = "Black" if is_white else "White"
        for attacker_start_pos in self.get_all_pieces(board, opponent):
            if self.is_legal_move(board, attacker_start_pos, king_pos, opponent):
                return True
        return False
    
    
    def find_king(self, board, king):
        for x in range(8):
            for y in range(8):
                piece = board[x][y]
                if piece == king:
                    return (x,y)
        print("king not found")
        

    def get_all_pieces(self,board, agent):
        #self.render_board(board)
        pieces = []
        for x in range(8):
            for y in range(8):
                piece = board[x][y]
                if (agent.lower() == "white" and piece.isupper()) or (agent.lower() == "black" and piece.islower()):
                    pieces.append((x,y))
        return pieces
    
    def get_possible_moves(self, board, piece, position):
        moves = []
        x, y = position

        # Pawn moves
        if piece == WHITE_PAWN:
            moves += self.get_pawn_moves(board,position, direction=-1, is_white=True)
        elif piece == BLACK_PAWN:
            moves += self.get_pawn_moves(board,position, direction=1, is_white=False)

        # Rook moves
        elif piece in [WHITE_ROOK, BLACK_ROOK]:
            moves += self.get_straight_line_moves(board,position, directions=[(1, 0), (-1, 0), (0, 1), (0, -1)])

        # Bishop moves
        elif piece in [WHITE_BISHOP, BLACK_BISHOP]:
            moves += self.get_straight_line_moves(board,position, directions=[(1, 1), (-1, -1), (1, -1), (-1, 1)])

        # Queen moves (combines rook and bishop)
        elif piece in [WHITE_QUEEN, BLACK_QUEEN]:
            moves += self.get_straight_line_moves(board,position, directions=[
                (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)
            ])

        # King moves (one square in any direction)
        elif piece in [WHITE_KING, BLACK_KING]:
            moves += self.get_king_moves(board,position)

        # Knight moves (L-shapes)
        elif piece in [WHITE_KNIGHT, BLACK_KNIGHT]:
            moves += self.get_knight_moves(board,position)

        return moves

    def get_pawn_moves(self, board, position, direction, is_white):
        x, y = position
        moves = []

        # Move forward one square
        if 0 <= x + direction < 8 and board[x + direction][y] == EMPTY:
            moves.append((x + direction, y))
            # Move forward two squares if on the starting rank
            if (x == 6 and is_white) or (x == 1 and not is_white):
                if board[x + 2 * direction][y] == EMPTY:
                    moves.append((x + 2 * direction, y))

        # Capture diagonally
        for dy in [-1, 1]:
            if 0 <= y + dy < 8 and 0 <= x + direction < 8:
                target = board[x + direction][y + dy]
                if (is_white and target.islower()) or (not is_white and target.isupper()):
                    moves.append((x + direction, y + dy))

        # En passant
        for dy in [-1, 1]:
            if self.is_en_passant_capture(board,position, (x + direction, y + dy), "White" if is_white else "Black"):
                moves.append((x + direction, y + dy))

        return moves

    def get_straight_line_moves(self, board, position, directions):
        moves = []
        for dx, dy in directions:
            x, y = position
            while True:
                x += dx
                y += dy
                if not (0 <= x < 8 and 0 <= y < 8):
                    break
                if board[x][y] == EMPTY:
                    moves.append((x, y))
                elif (board[x][y].islower() and board[position[0]][position[1]].isupper()) or \
                     (board[x][y].isupper() and board[position[0]][position[1]].islower()):
                    moves.append((x, y))
                    break
                else:
                    break
        return moves

    def get_king_moves(self, board,position):
        moves = []
        x, y = position
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                if board[new_x][new_y] == EMPTY or \
                   (board[new_x][new_y].islower() and board[x][y].isupper()) or \
                   (board[new_x][new_y].isupper() and board[x][y].islower()):
                    moves.append((new_x, new_y))
        # Castling moves
        if self.is_castle(board,"White" if board[x][y] == WHITE_KING else "Black", position, (x, y + 2)):
            moves.append((x, y + 2))
        if self.is_castle(board,"White" if board[x][y] == WHITE_KING else "Black", position, (x, y - 2)):
            moves.append((x, y - 2))
        return moves

    def get_knight_moves(self, board,position):
        moves = []
        x, y = position
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        for dx, dy in knight_moves:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                target = board[new_x][new_y]
                # Only allow move if space is empty or has an opponent piece
                if target == EMPTY or (target.islower() if board[x][y].isupper() else target.isupper()):
                    moves.append((new_x, new_y))
        return moves
    
    def is_stalemate(self, board,agent):
        if self.is_in_check(board,agent):
            return False
        pieces = self.get_all_pieces(board,agent)
        for start_pos in pieces:
            possible_moves = self.get_possible_moves(board,board[start_pos[0]][start_pos[1]], start_pos)
            for end_pos in possible_moves:
                if self.is_legal_move(board, start_pos, end_pos, agent) and not self.is_in_check_after_move(board, start_pos, end_pos, agent):
                    return False
        return True
    
    
        
    def results(self, board, action, agent, zobrist_hash=None,temporary=False):
        new_board = [row[:] for row in board]
        start_pos,end_pos = action
        if not self.is_legal_move(new_board, start_pos, end_pos, agent):
           raise ValueError(f"Illegal move by {agent}: {action}")  # Indicate an illegal move for debugging
        
        piece = new_board[start_pos[0]][start_pos[1]]
        captured = new_board[end_pos[0]][end_pos[1]]

        if zobrist_hash is not None:
            zobrist_hash = self.zobrist.update_hash(zobrist_hash,piece,start_pos)
            if captured != EMPTY:
                zobrist_hash = self.zobrist.update_hash(zobrist_hash, captured, end_pos)

        #record move
        if not temporary:
            self.actions_history[agent].append((*start_pos, *end_pos))
            
        # Handle en passant
        if self.is_en_passant_capture(new_board, start_pos, end_pos, agent):
            x1, y1 = end_pos
            ep_pos = (x1 - 1, y1) if agent == "Black" else (x1 + 1, y1)
            captured_piece = new_board[ep_pos[0]][ep_pos[1]]
            new_board[ep_pos[0]][ep_pos[1]] = EMPTY
            if zobrist_hash is not None:
                zobrist_hash = self.zobrist.update_hash(zobrist_hash, captured_piece, ep_pos)

        # Handle castling
        elif self.is_castle(new_board,agent, start_pos, end_pos):
            rook_start = ()
            rook_end = ()
            if agent == "Black":
                rook = BLACK_ROOK
                if end_pos == (0, 6):  # Kingside castling
                    rook_start = (0,7)
                    rook_end = (0,5)
                    new_board[0][5] = BLACK_ROOK
                    new_board[0][7] = EMPTY
                elif end_pos == (0, 2):  # Queenside castling
                    rook_start = (0,0)
                    rook_end = (0,3)
                    new_board[0][3] = BLACK_ROOK
                    new_board[0][0] = EMPTY
            elif agent == "White":
                rook = WHITE_ROOK
                if end_pos == (7, 6):  # Kingside castling
                    rook_start = (7,7)
                    rook_end = (7,5)
                    new_board[7][5] = WHITE_ROOK
                    new_board[7][7] = EMPTY
                elif end_pos == (7, 2):  # Queenside castling
                    rook_start = (7,0)
                    rook_end = (7,3)
                    new_board[7][3] = WHITE_ROOK
                    new_board[7][0] = EMPTY
            if zobrist_hash is not None:
                zobrist_hash = self.zobrist.update_hash(zobrist_hash, rook, rook_start)
                zobrist_hash = self.zobrist.update_hash(zobrist_hash,rook,rook_end)

        # Handle promotion
        elif self.is_promotion(new_board, start_pos, end_pos, agent):
            promoted_piece = WHITE_QUEEN if agent == "White" else BLACK_QUEEN
            new_board[end_pos[0]][end_pos[1]] = promoted_piece
            new_board[start_pos[0]][start_pos[1]] = EMPTY
            if zobrist_hash is not None:
                zobrist_hash = self.zobrist.update_hash(zobrist_hash, promoted_piece, end_pos)
            return new_board, zobrist_hash

        # Make the move permanently
        new_board[end_pos[0]][end_pos[1]] = piece
        new_board[start_pos[0]][start_pos[1]] = EMPTY
        if zobrist_hash is not None:
            zobrist_hash = self.zobrist.update_hash(zobrist_hash, piece, end_pos)
        return new_board,zobrist_hash

    def evaluate(self, board, agent):
        reward = 0
        opponent = "White" if agent.lower() == "black" else "Black"
        center_squares = [(3,3),(3,4),(4,3),(4,4)]
        player_pieces = self.get_all_pieces(board, agent)
        opponent_pieces = self.get_all_pieces(board, opponent)
        
        piece_rewards = {
        BLACK_PAWN: REWARD_MAP["pawn"],
        BLACK_ROOK: REWARD_MAP["rook"],
        BLACK_KNIGHT: REWARD_MAP["knight"],
        BLACK_BISHOP: REWARD_MAP["bishop"],
        BLACK_QUEEN: REWARD_MAP["queen"],
        WHITE_PAWN: REWARD_MAP["pawn"],
        WHITE_ROOK: REWARD_MAP["rook"],
        WHITE_KNIGHT: REWARD_MAP["knight"],
        WHITE_BISHOP: REWARD_MAP["bishop"],
        WHITE_QUEEN: REWARD_MAP["queen"]
        }
        #get kings position
        if agent.lower()=="white":
            king = WHITE_KING
            opponent_king = BLACK_KING
        else:
            king = BLACK_KING
            opponent_king = WHITE_KING

        king_pos = self.find_king(board,king)
        if king_pos is None:
            raise ValueError(f"King ({king}) not found for agent {agent} on board:\n{board}")

        opponent_king_pos = self.find_king(board, opponent_king)
        if opponent_king_pos is None:
            raise ValueError(f"Opponent King ({opponent_king}) not found for agent {agent} on board:\n{board}")

        for piece_pos in player_pieces:
            piece = board[piece_pos[0]][piece_pos[1]]
            reward += piece_rewards.get(piece, 0)
        for piece_pos in opponent_pieces:
            piece = board[piece_pos[0]][piece_pos[1]]
            reward -= piece_rewards.get(piece, 0)
        
        reward += sum(5 for pos in player_pieces if pos in center_squares)
        reward -= sum(5 for pos in opponent_pieces if pos in center_squares)
        
        #mobility
        agent_moves = len(self.actions(board,agent))
        opponent_moves = len(self.actions(board,opponent))
        total_pieces = len(player_pieces)+len(opponent_pieces)
        #reward based on state of game
        if total_pieces > 20:  # Opening
            reward += 0.1 * (agent_moves - opponent_moves)
        elif total_pieces > 10:  # Middlegame
            reward += 0.2 * (agent_moves - opponent_moves)
        else:  # Endgame
            reward += 10 * (len(player_pieces) - len(opponent_pieces))  # Material matters more
            reward += 5 * (8 - abs(king_pos[0] - 4) - abs(king_pos[1] - 4))
                
        #king saftey
        
        if self.is_king_exposed(board,king_pos):
            reward -= 15 
        if self.is_king_exposed(board,opponent_king_pos):
            reward +=15
        
        #pawn structure
        reward += self.evaluate_pawn_structure(board, agent) - self.evaluate_pawn_structure(board, opponent)
        
        #state checks
        if self.is_repetition(board):
            reward = -90
            return reward
        elif self.is_stalemate(board,agent):
            reward-=10
        if self.is_in_check(board, agent):
            reward+=REWARD_MAP["check_self"]
            if self.is_checkmate(board,agent):
                reward+=REWARD_MAP["loss"]
                return reward
        elif self.is_in_check(board,opponent):
            reward+=REWARD_MAP["check_opponent"]
            if self.is_checkmate(board, agent):
                reward+= REWARD_MAP["win"]
                return reward

        
        return round(reward,1)

    def evaluate_pawn_structure(self,board,agent):
        reward = 0
        pieces = self.get_all_pieces(board,agent)
        pawns = []
        for piece in pieces:
            if piece == BLACK_PAWN or piece == WHITE_PAWN:
                pawns.append(piece)
        for pawn in pawns:
            if self.is_lonely_pawn(board,pawn,agent):
                reward-=5
            if self.is_doubled_pawn(board,pawn,agent):
                reward-=5
            if self.is_connected_pawn(board,pawn,agent):
                reward +=5
        return reward
    
    def is_king_exposed(self,board,king_pos):
        surrounding_squares = self.get_surrounding_king_squares(king_pos)
        for square in surrounding_squares:
            if board[square[0]][square[1]] is EMPTY:
                return True
        return False
    
    def get_surrounding_king_squares(self,king_pos):
        x, y = king_pos
        squares = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                squares.append((new_x,new_y))
        return squares
    
    def is_lonely_pawn(self,board,pawn_pos,agent):
        row, col = pawn_pos
        adjacent_tiles=[col-1,col+1]
        for column in adjacent_tiles:
            if 0<= column < 8:
                for r in range(8):
                    piece = board[r][column]
                    if piece == board[row][col]:
                        return False
        return True
    
    def is_doubled_pawn(self,board,pawn_pos,agent):
        row,col = pawn_pos
        for r in range(8):
            if r != row:
                if board[r][col] == board[row][col]:
                    return True
        return False
                
    def is_connected_pawn(self,board,pawn_pos, agent):
        row,col = pawn_pos
        diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx,dy in diagonals:
            r,c = row+dx, col+dy
            if 0<= r<8 and 0<=c <8:
                if board[r][c] == board[row][col]:
                    return True
        return False
            
    def get_history(self):
        return {agent: list(moves) for agent, moves in self.actions_history.items()}
    
    def set_history(self,history):
        self.actions_history = {agent: list(moves) for agent, moves in history.items()}
        
    def is_tactical_action(self,action,board,agent):
        start_pos, end_pos = action
        piece = board[start_pos[0]][start_pos[1]]
        target = board[end_pos[0]][end_pos[1]]
        
        if target is not EMPTY:
            return True
        if self.is_in_check_after_move(board,start_pos,end_pos,agent):
            return True
        if self.is_promotion(board,start_pos,end_pos,agent):
            return True
        # if self.is_threatening_move(board,action,agent):
        #     return True
        return False
    
    def is_threatening_move(self,board,action,agent):
        start_pos,end_pos = action
        next_board = self.results(board,action,agent)
        opponent = "White" if agent.lower() == "black" else "Black"
        #simulate a is_check scenario on a piece
        opponent_pieces =self.get_all_pieces(next_board,opponent)
        for piece_pos in opponent_pieces:
            if self.is_legal_move(next_board,piece_pos,end_pos,opponent):
                return True
        return False
    
    def piece_value(self,piece):
        piece_rewards = {
        BLACK_PAWN: REWARD_MAP["pawn"],
        BLACK_ROOK: REWARD_MAP["rook"],
        BLACK_KNIGHT: REWARD_MAP["knight"],
        BLACK_BISHOP: REWARD_MAP["bishop"],
        BLACK_QUEEN: REWARD_MAP["queen"],
        "k":1000,
        WHITE_PAWN: REWARD_MAP["pawn"],
        WHITE_ROOK: REWARD_MAP["rook"],
        WHITE_KNIGHT: REWARD_MAP["knight"],
        WHITE_BISHOP: REWARD_MAP["bishop"],
        WHITE_QUEEN: REWARD_MAP["queen"],
        "K":1000
        
        }
        return piece_rewards[piece]
    
class Zobrist:
    def __init__(self):
        self.random_table = {
            piece: [random.getrandbits(64) for _ in range(64)]
            for piece in [
                BLACK_PAWN, BLACK_ROOK, BLACK_KNIGHT, BLACK_BISHOP, BLACK_QUEEN, BLACK_KING,
                WHITE_PAWN, WHITE_ROOK, WHITE_KNIGHT, WHITE_BISHOP, WHITE_QUEEN, WHITE_KING, EMPTY
            ]
        }
    
    def calculate_initial_hash(self,board):
        zobrist_hash = 0
        for x in range(8):
            for y in range(8):
                piece = board[x][y]
                if piece != EMPTY:
                    zobrist_hash ^= self.random_table[piece][x*8+y]
        return zobrist_hash
    
    def update_hash(self, zobrist_hash, piece, position):
        index = position[0] * 8 + position[1]
        zobrist_hash^=self.random_table[piece][index]
        return zobrist_hash