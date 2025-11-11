"""
PPO Agent Server - Case Closed Challenge
Uses pre-trained phase4 weights with opponent modeling
Flask-based server compatible with Judge Engine API
"""

import os
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import random

# Flask app setup
app = Flask(__name__)

# Agent identity
PARTICIPANT = os.getenv("PARTICIPANT", "PPO_Team")
AGENT_NAME = os.getenv("AGENT_NAME", "PPO_Phase4_Agent")

# Game state tracking
LAST_POSTED_STATE = {}
game_lock = Lock()


# ============================================================================
# PPO NEURAL NETWORK (Phase4 Compatible)
# ============================================================================

class TronPPOModel(nn.Module):
    """
    PPO Actor-Critic with opponent modeling head.
    Compatible with pre-trained phase4 weights.
    """
    
    def __init__(self, input_channels=7, grid_h=18, grid_w=20):
        super().__init__()
        
        # Convolutional feature extractor (matches phase4)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Flattened size: 32 * 18 * 20 = 11520
        self.flat_size = 32 * grid_h * grid_w
        
        # Policy head (8 actions in phase4)
        self.pi = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )
        
        # Value head
        self.v = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Opponent modeling head (trainable)
        self.opp_model = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
    
    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        
        logits = self.pi(h)
        value = self.v(h)
        opp_logits = self.opp_model(h)
        
        return logits, value, torch.softmax(opp_logits, dim=1)


# ============================================================================
# CHAMBER DETECTOR
# ============================================================================

class ChamberDetector:
    """Detects if agents are in separate chambers."""
    
    def __init__(self):
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_separated(self, board, my_pos, opp_pos):
        """BFS to check if positions are reachable."""
        visited = set([my_pos])
        queue = deque([my_pos])
        
        while queue:
            curr = queue.popleft()
            if curr == opp_pos:
                return False
            
            for dy, dx in self.directions:
                ny, nx = curr[0] + dy, curr[1] + dx
                if (0 <= ny < 18 and 0 <= nx < 20 and
                    (ny, nx) not in visited and board[0, ny, nx] == 0):
                    visited.add((ny, nx))
                    queue.append((ny, nx))
        
        return True


# ============================================================================
# PPO AGENT
# ============================================================================

class PPOAgent:
    """PPO Agent with phase4 weights and modular strategies."""
    
    def __init__(self, model_path='ppo_phase4.pt'):
        self.grid_height = 18
        self.grid_width = 20
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.direction_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        # Action mapping (8 actions -> 4 directions)
        self.action_mapping = {
            0: 0, 1: 1, 2: 2, 3: 3,
            4: 0, 5: 1, 6: 2, 7: 3
        }
        
        self.device = torch.device('cpu')
        self.model = TronPPOModel(
            input_channels=7,
            grid_h=self.grid_height,
            grid_w=self.grid_width
        ).to(self.device)
        
        self.chamber = ChamberDetector()
        
        # Load pre-trained weights
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"Warning: {model_path} not found, using random weights")
    
    def load_model(self, model_path):
        """Load pre-trained phase4 weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            print(f"✓ Loaded phase4 weights from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def convert_to_7channel(self, board_3ch, my_pos, opp_pos):
        """Convert 3-channel board to 7-channel format."""
        h, w = board_3ch.shape[1], board_3ch.shape[2]
        board_7ch = np.zeros((7, h, w), dtype=np.float32)
        
        # Original 3 channels
        board_7ch[0:3] = board_3ch
        
        # My position (one-hot)
        board_7ch[3, my_pos[0], my_pos[1]] = 1.0
        
        # Opponent position (one-hot)
        board_7ch[4, opp_pos[0], opp_pos[1]] = 1.0
        
        # Distance map
        for y in range(h):
            for x in range(w):
                dist = abs(y - my_pos[0]) + abs(x - my_pos[1])
                board_7ch[5, y, x] = 1.0 / (1.0 + dist)
        
        # Danger map
        for y in range(h):
            for x in range(w):
                if board_3ch[0, y, x] == 0:
                    danger = sum(
                        1 for dy, dx in self.directions
                        if (0 <= y + dy < h and 0 <= x + dx < w and
                            board_3ch[0, y + dy, x + dx] != 0)
                    )
                    board_7ch[6, y, x] = danger / 4.0
        
        return board_7ch
    
    def get_valid_moves(self, board, pos):
        """Get valid move indices."""
        valid = []
        for i, (dy, dx) in enumerate(self.directions):
            ny, nx = pos[0] + dy, pos[1] + dx
            if (0 <= ny < self.grid_height and 0 <= nx < self.grid_width and
                board[0, ny, nx] == 0):
                valid.append(i)
        return valid if valid else [0]
    
    def predict_opponent(self, board_3ch, my_pos, opp_pos):
        """Predict opponent's next move."""
        board_7ch = self.convert_to_7channel(board_3ch, my_pos, opp_pos)
        state = torch.tensor(board_7ch, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, _, opp_probs = self.model(state)
        
        return opp_probs.cpu().numpy()[0]
    
    def policy_action(self, board_3ch, pos, my_pos, opp_pos):
        """PPO policy inference."""
        board_7ch = self.convert_to_7channel(board_3ch, my_pos, opp_pos)
        state = torch.tensor(board_7ch, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, _, _ = self.model(state)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Map 8 actions to 4 directions
        probs_4 = np.zeros(4)
        for i in range(8):
            probs_4[self.action_mapping[i]] += probs[i]
        
        valid = self.get_valid_moves(board_3ch, pos)
        
        # Mask invalid actions
        mask = np.zeros(4)
        mask[valid] = 1.0
        masked_probs = probs_4 * mask
        
        if masked_probs.sum() == 0:
            return random.choice(valid)
        
        masked_probs /= masked_probs.sum()
        return np.random.choice(4, p=masked_probs)
    
    def mcts(self, board, my_pos, opp_pos, valid_moves, n_sims=150):
        """MCTS with opponent modeling."""
        move_scores = {i: 0 for i in valid_moves}
        opp_probs = self.predict_opponent(board, my_pos, opp_pos)
        
        for i in valid_moves:
            score = 0
            for _ in range(n_sims):
                opp_move = np.random.choice(4, p=opp_probs)
                
                new_my = (my_pos[0] + self.directions[i][0],
                         my_pos[1] + self.directions[i][1])
                new_opp = (opp_pos[0] + self.directions[opp_move][0],
                          opp_pos[1] + self.directions[opp_move][1])
                
                my_live = (0 <= new_my[0] < self.grid_height and
                          0 <= new_my[1] < self.grid_width and
                          board[0, new_my[0], new_my[1]] == 0)
                
                opp_live = (0 <= new_opp[0] < self.grid_height and
                           0 <= new_opp[1] < self.grid_width and
                           board[0, new_opp[0], new_opp[1]] == 0)
                
                if my_live and not opp_live:
                    score += 2
                elif my_live and opp_live:
                    score += 1
                elif not my_live and not opp_live:
                    score += 0.5
            
            move_scores[i] = score
        
        return max(move_scores.items(), key=lambda x: x[1])[0]
    
    def wall_following(self, board, pos, valid_moves):
        """Wall-following heuristic."""
        best_move = valid_moves[0]
        max_walls = -1
        
        for i in valid_moves:
            ny, nx = pos[0] + self.directions[i][0], pos[1] + self.directions[i][1]
            
            wall_count = sum(
                1 for dy, dx in self.directions
                if (0 <= ny + dy < self.grid_height and
                    0 <= nx + dx < self.grid_width and
                    board[0, ny + dy, nx + dx] != 0)
            )
            
            if wall_count > max_walls:
                best_move = i
                max_walls = wall_count
        
        return best_move
    
    def decide_move(self, board, my_pos, opp_pos, turn_number):
        """Main decision function with phase-aware strategies."""
        valid_moves = self.get_valid_moves(board, my_pos)
        
        if not valid_moves:
            return 'UP'
        
        if len(valid_moves) == 1:
            return self.direction_names[valid_moves[0]]
        
        # Phase detection
        separated = self.chamber.is_separated(board, my_pos, opp_pos)
        distance = abs(my_pos[0] - opp_pos[0]) + abs(my_pos[1] - opp_pos[1])
        close = distance < 6
        
        # Strategy selection
        if separated:
            move_idx = self.wall_following(board, my_pos, valid_moves)
        elif close:
            move_idx = self.mcts(board, my_pos, opp_pos, valid_moves, n_sims=150)
        else:
            move_idx = self.policy_action(board, my_pos, my_pos, opp_pos)
        
        if move_idx not in valid_moves:
            move_idx = valid_moves[0]
        
        return self.direction_names[move_idx]


# ============================================================================
# GLOBAL AGENT INSTANCE
# ============================================================================

# Initialize agent with phase4 weights
agent = PPOAgent(model_path='ppo_phase4.pt')


# ============================================================================
# FLASK API ENDPOINTS
# ============================================================================

@app.route("/", methods=["GET"])
def info():
    """Health/info endpoint."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Receive game state from judge."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)
    
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Return agent's move for current turn."""
    player_number = request.args.get("player_number", default=1, type=int)
    
    with game_lock:
        state = dict(LAST_POSTED_STATE)
        raw_board = state.get("board", [])
        
        if player_number == 1:
            my_trail = state.get("agent1_trail", [])
            opp_trail = state.get("agent2_trail", [])
        else:
            my_trail = state.get("agent2_trail", [])
            opp_trail = state.get("agent1_trail", [])
        
        turn_count = state.get("turn_count", 0)
    
    # Convert board to 3-channel format
    if len(my_trail) > 0 and len(opp_trail) > 0 and len(raw_board) > 0:
        try:
            board = np.zeros((3, 18, 20), dtype=np.float32)
            
            # Channel 0: walls/obstacles
            for y in range(len(raw_board)):
                for x in range(len(raw_board[0])):
                    if raw_board[y][x] != 0:
                        board[0, y, x] = 1
            
            # Channel 1: my trail
            for pos in my_trail:
                if len(pos) == 2:
                    x, y = pos
                    if 0 <= y < 18 and 0 <= x < 20:
                        board[1, y, x] = 1
            
            # Channel 2: opponent trail
            for pos in opp_trail:
                if len(pos) == 2:
                    x, y = pos
                    if 0 <= y < 18 and 0 <= x < 20:
                        board[2, y, x] = 1
            
            # Get positions (convert from (x, y) to (y, x))
            x_my, y_my = my_trail[-1]
            x_opp, y_opp = opp_trail[-1]
            my_pos = (y_my, x_my)
            opp_pos = (y_opp, x_opp)
            
            # Get move from PPO agent
            move = agent.decide_move(board, my_pos, opp_pos, turn_count)
        
        except Exception as e:
            print(f"Error in decision: {e}")
            move = "RIGHT"
    else:
        move = "RIGHT"
    
    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Acknowledge game end."""
    return jsonify({"status": "acknowledged"}), 200


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    print("="*70)
    print(f"PPO Agent Server Starting on port {port}")
    print("="*70)
    app.run(host="0.0.0.0", port=port, debug=False)
