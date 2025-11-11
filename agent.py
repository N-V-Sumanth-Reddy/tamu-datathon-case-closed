import os
import time
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
# ALPHA-BETA MINIMAX MODULE WITH ITERATIVE DEEPENING
# ============================================================================
class AlphaBetaMinimax:
    """Enhanced Minimax with iterative deepening and better evaluation."""
    
    def __init__(self, grid_width: int, grid_height: int, max_depth: int = 20):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_depth = max_depth
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.killer_moves = {}  # Killer move heuristic
        self.time_limit = 0.0
        self.start_time = 0.0
    
    def set_time_limit(self, seconds: float):
        """Set time limit for search."""
        self.start_time = time.time()
        self.time_limit = self.start_time + seconds
    
    def time_up(self) -> bool:
        """Check if time limit exceeded."""
        return time.time() >= self.time_limit
    
    def evaluate_advanced(self, board: np.ndarray, my_pos: Tuple[int, int],
                         opp_pos: Tuple[int, int]) -> float:
        """Advanced evaluation with territory, edges, and mobility."""
        # Voronoi territory
        def bfs_territory(start, opponent_pos):
            visited = set([start])
            queue = deque([start])
            edges = 0
            articulation_risk = 0
            
            while queue:
                y, x = queue.popleft()
                neighbors = 0
                for dy, dx in self.directions:
                    ny = (y + dy) % self.grid_height
                    nx = (x + dx) % self.grid_width
                    
                    if board[ny, nx] == 0 and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append((ny, nx))
                        neighbors += 1
                    elif board[ny, nx] == 0:
                        neighbors += 1
                
                # Count edges (connections to empty spaces)
                edges += neighbors
                
                # Articulation point risk (positions with few neighbors)
                if neighbors <= 1:
                    articulation_risk += 1
            
            return len(visited), edges, articulation_risk
        
        my_area, my_edges, my_risk = bfs_territory(my_pos, opp_pos)
        opp_area, opp_edges, opp_risk = bfs_territory(opp_pos, my_pos)
        
        # Weighted evaluation
        # Territory: 50 points per square
        # Edges: 10 points per edge (mobility)
        # Risk: -20 points per articulation point
        my_score = 50 * my_area + 10 * my_edges - 20 * my_risk
        opp_score = 50 * opp_area + 10 * opp_edges - 20 * opp_risk
        
        return my_score - opp_score
    
    def get_valid_moves(self, board: np.ndarray, pos: Tuple[int, int]) -> List[int]:
        """Get valid move indices with torus wraparound."""
        valid = []
        for idx, (dy, dx) in enumerate(self.directions):
            ny, nx = pos[0] + dy, pos[1] + dx
            # Apply torus wraparound
            ny = ny % self.grid_height
            nx = nx % self.grid_width
            if board[ny, nx] == 0:
                valid.append(idx)
        return valid
    
    def minimax(self, board: np.ndarray, my_pos: Tuple[int, int],
               opp_pos: Tuple[int, int], depth: int, alpha: float, beta: float,
               maximizing: bool, ply: int = 0) -> Tuple[float, Optional[int]]:
        """Minimax with alpha-beta pruning and killer moves."""
        # Check time limit
        if self.time_up():
            return self.evaluate_advanced(board, my_pos, opp_pos), None
        
        if depth == 0:
            return self.evaluate_advanced(board, my_pos, opp_pos), None
        
        pos = my_pos if maximizing else opp_pos
        valid_moves = self.get_valid_moves(board, pos)
        
        if not valid_moves:
            return (-10000 if maximizing else 10000), None
        
        best_move = valid_moves[0]
        
        if maximizing:
            max_eval = float('-inf')
            
            # Killer move heuristic - try best move from previous search first
            if ply in self.killer_moves:
                killer = self.killer_moves[ply]
                if killer in valid_moves:
                    valid_moves.remove(killer)
                    valid_moves.insert(0, killer)
            
            for move_idx in valid_moves:
                new_board = board.copy()
                dy, dx = self.directions[move_idx]
                new_my_pos = ((my_pos[0] + dy) % self.grid_height, 
                             (my_pos[1] + dx) % self.grid_width)
                new_board[new_my_pos] = 2
                
                eval_score, _ = self.minimax(new_board, new_my_pos, opp_pos,
                                            depth - 1, alpha, beta, False, ply + 1)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move_idx
                    self.killer_moves[ply] = move_idx  # Store killer move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        else:
            min_eval = float('inf')
            
            # Killer move heuristic for opponent
            if ply in self.killer_moves:
                killer = self.killer_moves[ply]
                if killer in valid_moves:
                    valid_moves.remove(killer)
                    valid_moves.insert(0, killer)
            
            for move_idx in valid_moves:
                new_board = board.copy()
                dy, dx = self.directions[move_idx]
                new_opp_pos = ((opp_pos[0] + dy) % self.grid_height,
                              (opp_pos[1] + dx) % self.grid_width)
                new_board[new_opp_pos] = 3
                
                eval_score, _ = self.minimax(new_board, my_pos, new_opp_pos,
                                            depth - 1, alpha, beta, True, ply + 1)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move_idx
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_move
    
    def get_best_move_iterative(self, board: np.ndarray, my_pos: Tuple[int, int],
                               opp_pos: Tuple[int, int], time_limit: float = 0.7) -> int:
        """Iterative deepening search - searches deeper until time runs out."""
        self.set_time_limit(time_limit)
        self.killer_moves.clear()
        
        best_move = 0
        best_value = float('-inf')
        
        # Iterative deepening: start shallow, go deeper
        for depth in range(1, self.max_depth + 1):
            if self.time_up():
                break
            
            value, move = self.minimax(board, my_pos, opp_pos, depth,
                                      float('-inf'), float('inf'), True, 0)
            
            if not self.time_up() and move is not None:
                best_move = move
                best_value = value
                
                # If we found a winning move, stop searching
                if value > 9000:
                    break
            else:
                # Timeout during this depth, use previous result
                break
        
        return best_move if best_move is not None else 0
    
    def get_best_move(self, board: np.ndarray, my_pos: Tuple[int, int],
                     opp_pos: Tuple[int, int]) -> int:
        """Find best move using iterative deepening."""
        return self.get_best_move_iterative(board, my_pos, opp_pos, 0.7)


# ============================================================================
# MCTS MODULE
# ============================================================================
class MCTS:
    """Enhanced Monte-Carlo Tree Search for critical decisions."""
    
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
        return manhattan <= 4 or my_valid <= 2 or filled > 0.7
    
    def get_valid_moves(self, board: np.ndarray, pos: Tuple[int, int]) -> List[int]:
        """Get valid moves with torus wraparound."""
        valid = []
        for idx, (dy, dx) in enumerate(self.directions):
            ny, nx = pos[0] + dy, pos[1] + dx
            # Apply torus wraparound
            ny = ny % self.grid_height
            nx = nx % self.grid_width
            if board[ny, nx] == 0:
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
            
            current_my = ((current_my[0] + dy_m) % self.grid_height,
                         (current_my[1] + dx_m) % self.grid_width)
            current_opp = ((current_opp[0] + dy_o) % self.grid_height,
                          (current_opp[1] + dx_o) % self.grid_width)
            
            current_board[current_my] = 2
            current_board[current_opp] = 3
        
        return 0.5
    
    def search(self, board: np.ndarray, my_pos: Tuple[int, int],
              opp_pos: Tuple[int, int], n_sims: int = 500) -> int:
        """Enhanced MCTS search with more simulations."""
        my_valid = self.get_valid_moves(board, my_pos)
        if len(my_valid) <= 1:
            return my_valid[0] if my_valid else 0
        
        move_stats = {m: {'wins': 0, 'visits': 0} for m in my_valid}
        
        # More simulations for better accuracy
        for _ in range(n_sims):
            for my_move in my_valid:
                opp_valid = self.get_valid_moves(board, opp_pos)
                if not opp_valid:
                    continue
                
                opp_move = np.random.choice(opp_valid)
                new_board = board.copy()
                
                dy_m, dx_m = self.directions[my_move]
                dy_o, dx_o = self.directions[opp_move]
                
                new_my = ((my_pos[0] + dy_m) % self.grid_height,
                         (my_pos[1] + dx_m) % self.grid_width)
                new_opp = ((opp_pos[0] + dy_o) % self.grid_height,
                          (opp_pos[1] + dx_o) % self.grid_width)
                
                new_board[new_my] = 2
                new_board[new_opp] = 3
                
                result = self.simulate_playout(new_board, new_my, new_opp, max_steps=20)
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
        
        # Initialize enhanced modules
        self.chamber_detector = ChamberDetector(self.grid_width, self.grid_height)
        self.minimax = AlphaBetaMinimax(self.grid_width, self.grid_height, max_depth=20)
        self.mcts = MCTS(self.grid_width, self.grid_height)
        
        # Direction mapping
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.direction_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        # Game state
        self.move_count = 0
    
    def get_valid_moves(self, board: np.ndarray, pos: Tuple[int, int]) -> List[int]:
        """Get valid move indices with torus wraparound support."""
        valid = []
        for idx, (dy, dx) in enumerate(self.directions):
            ny, nx = pos[0] + dy, pos[1] + dx
            # Apply torus wraparound (modulo)
            ny = ny % self.grid_height
            nx = nx % self.grid_width
            # Check if space is empty
            if board[ny, nx] == 0:
                valid.append(idx)
        return valid
    
    def decide_move(self, board: np.ndarray, my_pos: Tuple[int, int],
                   opp_pos: Tuple[int, int], turn_number: int) -> str:
        """Main decision function."""
        valid_moves = self.get_valid_moves(board, my_pos)
        
        # Emergency: No valid moves - try any direction with wraparound
        if not valid_moves:
            # Try all directions with wraparound and pick first empty space
            for idx, (dy, dx) in enumerate(self.directions):
                ny = (my_pos[0] + dy) % self.grid_height
                nx = (my_pos[1] + dx) % self.grid_width
                if board[ny, nx] == 0:
                    return self.direction_names[idx]
            # Absolute last resort - completely trapped
            return 'RIGHT'
        
        if len(valid_moves) == 1:
            return self.direction_names[valid_moves[0]]
        
        # Strategy selection based on game state
        is_stationary = self.chamber_detector.is_stationary_phase(board, my_pos, opp_pos)
        is_critical = self.mcts.is_critical(board, my_pos, opp_pos)
        
        # Enhanced strategy with iterative deepening as primary
        if is_critical and turn_number > 15:
            # Use MCTS for very critical decisions (more simulations)
            move_idx = self.mcts.search(board, my_pos, opp_pos, n_sims=500)
        elif not is_stationary:
            # Use iterative deepening minimax (primary strategy)
            move_idx = self.minimax.get_best_move(board, my_pos, opp_pos)
        else:
            # Use wall-following heuristic in stationary phase
            move_idx = self.wall_following_heuristic(board, my_pos, valid_moves)
        
        # Ensure valid move - double check
        if move_idx not in valid_moves:
            move_idx = valid_moves[0]
        
        self.move_count += 1
        return self.direction_names[move_idx]
    
    def should_use_boost(self, board: np.ndarray, my_pos: Tuple[int, int],
                        opp_pos: Tuple[int, int], move_idx: int, 
                        boosts_remaining: int, turn_number: int) -> bool:
        """Decide whether to use a boost on this move."""
        if boosts_remaining <= 0:
            return False
        
        # Get the direction we're moving with wraparound
        dy, dx = self.directions[move_idx]
        first_step = ((my_pos[0] + dy) % self.grid_height,
                     (my_pos[1] + dx) % self.grid_width)
        second_step = ((first_step[0] + dy) % self.grid_height,
                      (first_step[1] + dx) % self.grid_width)
        
        # Check if both steps are empty
        if board[first_step] != 0:
            return False
        
        if board[second_step] != 0:
            return False
        
        # SAFETY CHECK: Don't boost if opponent is very close (might collide)
        manhattan_now = abs(my_pos[0] - opp_pos[0]) + abs(my_pos[1] - opp_pos[1])
        manhattan_first = abs(first_step[0] - opp_pos[0]) + abs(first_step[1] - opp_pos[1])
        manhattan_second = abs(second_step[0] - opp_pos[0]) + abs(second_step[1] - opp_pos[1])
        
        # Don't boost if we're getting too close to opponent (collision risk)
        if manhattan_first <= 2 or manhattan_second <= 2:
            return False
        
        # Count free spaces
        my_free_spaces = len(self.get_valid_moves(board, my_pos))
        free_spaces_after = 0
        for ndy, ndx in self.directions:
            ny, nx = second_step[0] + ndy, second_step[1] + ndx
            if (0 <= ny < self.grid_height and 0 <= nx < self.grid_width and
                board[ny, nx] == 0 and (ny, nx) != first_step):
                free_spaces_after += 1
        
        # Use boost in these scenarios:
        
        # 1. Early game (turns 5-15): Use boost to gain territory advantage
        if 5 <= turn_number <= 15 and boosts_remaining >= 2:
            if manhattan_second > manhattan_now + 1 and free_spaces_after >= 3:
                return True
        
        # 2. Mid game (turns 15-40): Use boost when it significantly increases distance
        if 15 <= turn_number <= 40 and boosts_remaining >= 1:
            if manhattan_second - manhattan_now >= 3 and free_spaces_after >= 2:
                return True
        
        # 3. Critical escape: Use boost to escape tight spots
        if my_free_spaces <= 2 and free_spaces_after >= 3 and manhattan_second > manhattan_now:
            return True
        
        # 4. Territory control: Use boost when we have space advantage
        if turn_number > 10 and boosts_remaining >= 2:
            # Calculate Voronoi-like territory estimate
            my_territory, opp_territory = self.chamber_detector.compute_voronoi(
                board, my_pos, opp_pos
            )
            
            # If we're ahead in territory, use boost to maintain lead
            if my_territory > opp_territory * 1.2 and free_spaces_after >= 3:
                return True
        
        # 5. Late game: Save last boost for emergency only
        if boosts_remaining == 1 and turn_number > 80:
            # Only use if we're in real danger
            if my_free_spaces == 1 and free_spaces_after >= 2:
                return True
            return False
        
        return False
    
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
                ny = (new_pos[0] + ndy) % self.grid_height
                nx = (new_pos[1] + ndx) % self.grid_width
                if board[ny, nx] == 0:
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
            boosts_remaining = state.get("agent1_boosts", 0)
        else:
            my_trail = state.get("agent2_trail", [])
            opp_trail = state.get("agent1_trail", [])
            boosts_remaining = state.get("agent2_boosts", 0)
        
        turn_count = state.get("turn_count", 0)
    
    # Get positions from trails
    # Note: Game uses (x, y) but agent uses (y, x) internally
    if len(my_trail) > 0 and len(opp_trail) > 0:
        # Convert from (x, y) to (y, x) for internal use
        my_pos = (my_trail[-1][1], my_trail[-1][0])  # (y, x)
        opp_pos = (opp_trail[-1][1], opp_trail[-1][0])  # (y, x)
        
        # Use advanced agent logic
        move = agent.decide_move(board, my_pos, opp_pos, turn_count)
        
        # Check if we should use boost
        move_idx = agent.direction_names.index(move)
        use_boost = agent.should_use_boost(board, my_pos, opp_pos, move_idx, 
                                           boosts_remaining, turn_count)
        
        if use_boost:
            move = f"{move}:BOOST"
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
