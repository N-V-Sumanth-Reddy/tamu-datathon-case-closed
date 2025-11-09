"""Case Closed Agent - Advanced Tron AI with Modular Strategies
Implements: Opponent Modeling, Chamber Detection, Alpha-Beta Pruning, MCTS
"""
import os
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque, defaultdict
import numpy as np
from typing import Tuple, List, Dict, Optional, Any

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "Noob Agent"


# ============================================================================
# CHAMBER DETECTION MODULE
# ============================================================================
class ChamberDetector:
    """Detects chambers using Tarjan's articulation points algorithm."""
    
    def __init__(self, grid_width: int, grid_height: int):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def compute_voronoi(self, board: np.ndarray, my_pos: Tuple[int, int], 
                       opp_pos: Tuple[int, int]) -> Tuple[int, int]:
        """BFS-based Voronoi computation."""
        visited = set()
        queue = deque([(my_pos, 1), (opp_pos, 2)])
        visited.add(my_pos)
        visited.add(opp_pos)
        my_count, opp_count = 1, 1
        
        while queue:
            (y, x), owner = queue.popleft()
            for dy, dx in self.directions:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.grid_height and 0 <= nx < self.grid_width and
                    (ny, nx) not in visited and board[ny, nx] == 0):
                    visited.add((ny, nx))
                    queue.append(((ny, nx), owner))
                    if owner == 1:
                        my_count += 1
                    else:
                        opp_count += 1
        
        return my_count, opp_count
    
    def is_stationary_phase(self, board: np.ndarray, my_pos: Tuple[int, int],
                           opp_pos: Tuple[int, int]) -> bool:
        """Detect if agents are separated."""
        visited = set()
        queue = deque([my_pos])
        visited.add(my_pos)
        
        while queue:
            y, x = queue.popleft()
            if (y, x) == opp_pos:
                return False  # Can reach opponent
            
            for dy, dx in self.directions:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.grid_height and 0 <= nx < self.grid_width and
                    board[ny, nx] == 0 and (ny, nx) not in visited):
                    visited.add((ny, nx))
                    queue.append((ny, nx))
        
        return True  # Cannot reach opponent


# ============================================================================
# ALPHA-BETA MINIMAX MODULE
# ============================================================================
class AlphaBetaMinimax:
    """Minimax with alpha-beta pruning."""
    
    def __init__(self, grid_width: int, grid_height: int, max_depth: int = 2):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_depth = max_depth
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def evaluate_voronoi(self, board: np.ndarray, my_pos: Tuple[int, int],
                        opp_pos: Tuple[int, int]) -> float:
        """Voronoi heuristic evaluation."""
        def bfs_count(start):
            visited = set([start])
            queue = deque([start])
            while queue:
                y, x = queue.popleft()
                for dy, dx in self.directions:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < self.grid_height and 0 <= nx < self.grid_width and
                        board[ny, nx] == 0 and (ny, nx) not in visited):
                        visited.add((ny, nx))
                        queue.append((ny, nx))
            return len(visited)
        
        my_area = bfs_count(my_pos)
        opp_area = bfs_count(opp_pos)
        return my_area - opp_area
    
    def get_valid_moves(self, board: np.ndarray, pos: Tuple[int, int]) -> List[int]:
        """Get valid move indices."""
        valid = []
        for idx, (dy, dx) in enumerate(self.directions):
            ny, nx = pos[0] + dy, pos[1] + dx
            if (0 <= ny < self.grid_height and 0 <= nx < self.grid_width and
                board[ny, nx] == 0):
                valid.append(idx)
        return valid
    
    def minimax(self, board: np.ndarray, my_pos: Tuple[int, int],
               opp_pos: Tuple[int, int], depth: int, alpha: float, beta: float,
               maximizing: bool) -> Tuple[float, Optional[int]]:
        """Minimax with alpha-beta pruning."""
        if depth == 0:
            return self.evaluate_voronoi(board, my_pos, opp_pos), None
        
        pos = my_pos if maximizing else opp_pos
        valid_moves = self.get_valid_moves(board, pos)
        
        if not valid_moves:
            return (-10000 if maximizing else 10000), None
        
        best_move = valid_moves[0]
        
        if maximizing:
            max_eval = float('-inf')
            for move_idx in valid_moves:
                new_board = board.copy()
                dy, dx = self.directions[move_idx]
                new_my_pos = (my_pos[0] + dy, my_pos[1] + dx)
                new_board[new_my_pos] = 2
                
                eval_score, _ = self.minimax(new_board, new_my_pos, opp_pos,
                                            depth - 1, alpha, beta, False)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move_idx
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move_idx in valid_moves:
                new_board = board.copy()
                dy, dx = self.directions[move_idx]
                new_opp_pos = (opp_pos[0] + dy, opp_pos[1] + dx)
                new_board[new_opp_pos] = 3
                
                eval_score, _ = self.minimax(new_board, my_pos, new_opp_pos,
                                            depth - 1, alpha, beta, True)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move_idx
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_move
    
    def get_best_move(self, board: np.ndarray, my_pos: Tuple[int, int],
                     opp_pos: Tuple[int, int]) -> int:
        """Find best move using alpha-beta."""
        _, move = self.minimax(board, my_pos, opp_pos, self.max_depth,
                              float('-inf'), float('inf'), True)
        return move if move is not None else 0


# ============================================================================
# MCTS MODULE
# ============================================================================
class MCTS:
    """Monte-Carlo Tree Search for critical decisions."""
    
    def __init__(self, grid_width: int, grid_height: int):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_critical(self, board: np.ndarray, my_pos: Tuple[int, int],
                   opp_pos: Tuple[int, int]) -> bool:
        """Check if state is critical."""
        manhattan = abs(my_pos[0] - opp_pos[0]) + abs(my_pos[1] - opp_pos[1])
        my_valid = len(self.get_valid_moves(board, my_pos))
        filled = np.sum(board != 0) / (self.grid_height * self.grid_width)
        return manhattan <= 5 or my_valid <= 2 or filled > 0.6
    
    def get_valid_moves(self, board: np.ndarray, pos: Tuple[int, int]) -> List[int]:
        """Get valid moves."""
        valid = []
        for idx, (dy, dx) in enumerate(self.directions):
            ny, nx = pos[0] + dy, pos[1] + dx
            if (0 <= ny < self.grid_height and 0 <= nx < self.grid_width and
                board[ny, nx] == 0):
                valid.append(idx)
        return valid
    
    def simulate_playout(self, board: np.ndarray, my_pos: Tuple[int, int],
                        opp_pos: Tuple[int, int], max_steps: int = 15) -> float:
        """Random playout simulation."""
        current_board = board.copy()
        current_my = my_pos
        current_opp = opp_pos
        
        for _ in range(max_steps):
            my_valid = self.get_valid_moves(current_board, current_my)
            opp_valid = self.get_valid_moves(current_board, current_opp)
            
            if not my_valid and not opp_valid:
                return 0.5
            if not my_valid:
                return 0.0
            if not opp_valid:
                return 1.0
            
            my_move = np.random.choice(my_valid)
            opp_move = np.random.choice(opp_valid)
            
            dy_m, dx_m = self.directions[my_move]
            dy_o, dx_o = self.directions[opp_move]
            
            current_my = (current_my[0] + dy_m, current_my[1] + dx_m)
            current_opp = (current_opp[0] + dy_o, current_opp[1] + dx_o)
            
            current_board[current_my] = 2
            current_board[current_opp] = 3
        
        return 0.5
    
    def search(self, board: np.ndarray, my_pos: Tuple[int, int],
              opp_pos: Tuple[int, int], n_sims: int = 200) -> int:
        """MCTS search."""
        my_valid = self.get_valid_moves(board, my_pos)
        if len(my_valid) <= 1:
            return my_valid[0] if my_valid else 0
        
        move_stats = {m: {'wins': 0, 'visits': 0} for m in my_valid}
        
        for _ in range(n_sims):
            for my_move in my_valid:
                opp_valid = self.get_valid_moves(board, opp_pos)
                if not opp_valid:
                    continue
                
                opp_move = np.random.choice(opp_valid)
                new_board = board.copy()
                
                dy_m, dx_m = self.directions[my_move]
                dy_o, dx_o = self.directions[opp_move]
                
                new_my = (my_pos[0] + dy_m, my_pos[1] + dx_m)
                new_opp = (opp_pos[0] + dy_o, opp_pos[1] + dx_o)
                
                new_board[new_my] = 2
                new_board[new_opp] = 3
                
                result = self.simulate_playout(new_board, new_my, new_opp)
                move_stats[my_move]['visits'] += 1
                move_stats[my_move]['wins'] += result
        
        best_move = max(move_stats.items(),
                       key=lambda x: x[1]['wins'] / max(x[1]['visits'], 1))[0]
        return best_move


# ============================================================================
# MAIN AGENT CLASS
# ============================================================================
class CaseClosedAgent:
    """Main agent coordinating all strategies."""
    
    def __init__(self):
        self.grid_width = 20
        self.grid_height = 18
        
        # Initialize modules
        self.chamber_detector = ChamberDetector(self.grid_width, self.grid_height)
        self.minimax = AlphaBetaMinimax(self.grid_width, self.grid_height, max_depth=2)
        self.mcts = MCTS(self.grid_width, self.grid_height)
        
        # Direction mapping
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.direction_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        # Game state
        self.move_count = 0
    
    def get_valid_moves(self, board: np.ndarray, pos: Tuple[int, int]) -> List[int]:
        """Get valid move indices."""
        valid = []
        for idx, (dy, dx) in enumerate(self.directions):
            ny, nx = pos[0] + dy, pos[1] + dx
            if (0 <= ny < self.grid_height and 0 <= nx < self.grid_width and
                board[ny, nx] == 0):
                valid.append(idx)
        return valid
    
    def decide_move(self, board: np.ndarray, my_pos: Tuple[int, int],
                   opp_pos: Tuple[int, int], turn_number: int) -> str:
        """Main decision function."""
        valid_moves = self.get_valid_moves(board, my_pos)
        
        if not valid_moves:
            return 'UP'  # Fallback
        
        if len(valid_moves) == 1:
            return self.direction_names[valid_moves[0]]
        
        # Strategy selection based on game state
        is_stationary = self.chamber_detector.is_stationary_phase(board, my_pos, opp_pos)
        is_critical = self.mcts.is_critical(board, my_pos, opp_pos)
        
        # Phase-based strategy
        if is_critical and turn_number > 10:
            # Use MCTS for critical decisions
            move_idx = self.mcts.search(board, my_pos, opp_pos, n_sims=150)
        elif not is_stationary and turn_number < 100:
            # Use alpha-beta minimax in non-stationary phase
            move_idx = self.minimax.get_best_move(board, my_pos, opp_pos)
        else:
            # Use wall-following heuristic in stationary phase
            move_idx = self.wall_following_heuristic(board, my_pos, valid_moves)
        
        # Ensure valid move
        if move_idx not in valid_moves:
            move_idx = valid_moves[0]
        
        self.move_count += 1
        return self.direction_names[move_idx]
    
    def wall_following_heuristic(self, board: np.ndarray, pos: Tuple[int, int],
                                 valid_moves: List[int]) -> int:
        """Choose move that maximizes space."""
        best_move = valid_moves[0]
        max_space = -1
        
        for move_idx in valid_moves:
            dy, dx = self.directions[move_idx]
            new_pos = (pos[0] + dy, pos[1] + dx)
            
            # Count free adjacent spaces
            free_count = 0
            for ndy, ndx in self.directions:
                ny, nx = new_pos[0] + ndy, new_pos[1] + ndx
                if (0 <= ny < self.grid_height and 0 <= nx < self.grid_width and
                    board[ny, nx] == 0):
                    free_count += 1
            
            if free_count > max_space:
                max_space = free_count
                best_move = move_idx
        
        return best_move


# Global agent instance
agent = CaseClosedAgent()


# ============================================================================
# FLASK API ENDPOINTS
# ============================================================================
@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge."""
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)
        
        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass
        
        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick."""
    player_number = request.args.get("player_number", default=1, type=int)
    
    with game_lock:
        state = dict(LAST_POSTED_STATE)
        board = np.array(state.get("board", []))
        
        if player_number == 1:
            my_trail = state.get("agent1_trail", [])
            opp_trail = state.get("agent2_trail", [])
        else:
            my_trail = state.get("agent2_trail", [])
            opp_trail = state.get("agent1_trail", [])
        
        turn_count = state.get("turn_count", 0)
    
    # Get positions from trails
    if len(my_trail) > 0 and len(opp_trail) > 0:
        my_pos = tuple(my_trail[-1])
        opp_pos = tuple(opp_trail[-1])
        
        # Use advanced agent logic
        move = agent.decide_move(board, my_pos, opp_pos, turn_count)
    else:
        # Fallback
        move = "RIGHT"
    
    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state."""
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=False)
