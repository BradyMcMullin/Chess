# Chess

This is a custom chess environment that implements the game of chess for AI experimentation and gameplay.

| Import             | ```import chess```                            |
| ------------------ | --------------------------------------------- |
| Actions            |Discrete                                       | 
| Parallel API       | Yes                                           |
| Manual Control     | Yes                                           |
| Agents             | ```agents=['Black','White']```                |
| Action Shape       | MultiDiscrete([8,8,8,8])                      |
| Action Values      | MultiDiscrete([8,8,8,8])                      |
| Observation Shape  | (8,8,111)                                     |
| Observation Values | ['k','K','q','Q','b','B','n','N','r','R','.'] |


## Observation Space
The observation is a dictionary which contains a 'board' element which gives the current state of the chess board, a 'zobrist_hash' element which gives the current hash value of the board, and a 'model' element which holds logic for the chess board, which includes a actions method that returns a list of available actions.

### Example Board
```
[
    ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
    ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
]
```

## Action Space
The action space uses a MultiDiscrete([8, 8, 8, 8]) representation, where each action is defined by four integers:
- (8): Source Row - The rank of the piece to move (0-7)
- (8): Source Column – The file of the piece to move (0–7).
- (8): Target Row – The destination rank (0–7).
- (8): Target Column – The destination file (0–7).

### Example Action 
For a board with notation b1 -> a3 the action would be (7,1,5,0)

### Rules and Validity
The environment enforces standard chess rules, that are in seperate function in the model class including:
- Legal moves for each piece
- Castling, en passant, and pawn promotion
- Check and checkmate detection  
- Tie detection including stalemate, and threefold repetition

### Legal actions
Legal actions are included in functions from the model class. The main legal move function is the action function which takes in a board state and returns all possible actions.<br>
Example: <br>
```actions = observation["model"].actions(current_board)```

## Rewards
Rewards are based on two criteria which each have sub criteria.<br>

### Game state reward - Based on the models heuristic which include: <br>
- +- 1000 for a win/loss
- +- 50 for a check on player/opponent side
- +- 9 for each queen on player/opponent side
- +- 5 for each rook on player/opponent side
- +- 3 for each bishop on player/opponent side
- +- 3 for each night on player/opponent side
- +- 1 for each pawn on player/opponent side
- +- points for better gameplay such as center advantage etc... see code for more detail

### Win rewards:
- +1 for a win
- -1 for a loss
- 0 for a tie


## Usage

### AEC

```
import chess

env = chess.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        env.step(None) #used to give immediate loss
        continue
    else:
        board = observation["board"]
        actions = observation["model"].actions(board)
        action = agents[agent].agent_function(board, actions, agent)
    
    env.step(action)
env.close()
```
