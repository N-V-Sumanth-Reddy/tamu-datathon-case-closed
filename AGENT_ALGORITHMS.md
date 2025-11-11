# Agent 1 - Crash Avoidance Algorithms

## Overview
Agent 1 uses a **multi-layered approach** combining several AI algorithms for crash avoidance and strategic play.

---

## 1. Valid Move Filtering (Primary Defense)

**Purpose**: Prevent immediate crashes by filtering out invalid moves

**How it works**:
```python
def get_valid_moves(board, pos):
    - Check each direction (UP, DOWN, LEFT, RIGHT)
    - Validate: 0 <= new_y < 18 and 0 <= new_x < 20 (boundary check)
    - Validate: board[new_y][new_x] == 0 (no obstacle/trail)
    - Return only safe moves
```

**Prevents**:
- ✅ Boundary crashes (hitting walls)
- ✅ Trail collisions (hitting own/opponent trail)

---

## 2. Alpha-Beta Minimax (Strategic Planning)

**Purpose**: Look ahead and choose moves that maximize territory

**Algorithm**:
- **Depth**: 2 moves ahead
- **Evaluation**: Voronoi territory calculation (BFS-based)
- **Pruning**: Alpha-beta to reduce search space
- **Score**: my_territory - opponent_territory

**How it avoids crashes**:
- Simulates future board states
- Only considers valid moves at each depth
- Chooses path with maximum safe territory
- Returns -10000 score if no valid moves (dead end)

**Used when**: 
- Not in stationary phase (agents can reach each other)
- Turn < 100

---

## 3. Monte Carlo Tree Search (MCTS)

**Purpose**: Handle critical/dangerous situations with probabilistic search

**Algorithm**:
- Runs 150 random simulations per move
- For each valid move:
  - Simulate random playouts (15 steps)
  - Track wins/visits ratio
- Choose move with highest win rate

**Triggers when**:
- Manhattan distance to opponent ≤ 5 (close combat)
- Only 2 or fewer valid moves (tight spot)
- Board >60% filled (late game)

**How it avoids crashes**:
- Tests many random scenarios
- Learns which moves lead to survival
- Avoids moves that frequently lead to crashes in simulations

---

## 4. Wall-Following Heuristic

**Purpose**: Maximize available space in separated chambers

**Algorithm**:
```python
For each valid move:
    - Count free adjacent spaces at new position
    - Choose move with maximum open space
```

**Used when**: 
- Agents are separated (stationary phase)
- BFS cannot reach opponent

**How it avoids crashes**:
- Prefers moves with more escape routes
- Avoids corners and dead ends
- Maximizes future mobility

---

## 5. Chamber Detection (Voronoi Computation)

**Purpose**: Detect when agents are separated and calculate territory

**Algorithm**:
- Simultaneous BFS from both agent positions
- Mark cells by which agent reaches them first
- Count territory for each agent

**Benefits**:
- Identifies when to switch strategies
- Helps evaluate position strength
- Used in boost decision logic

---

## 6. Boost Strategy

**Purpose**: Use boosts strategically without causing crashes

**Safety Checks**:
1. ✅ Both boost steps must be valid (in bounds + empty)
2. ✅ Don't boost if it brings us within 2 spaces of opponent
3. ✅ Check free spaces after boost (need ≥2 escape routes)

**When to boost**:
- Early game (turns 5-15): Gain territory advantage
- Mid game (turns 15-40): Increase distance from opponent
- Escape tight spots: When current position has ≤2 valid moves
- Territory advantage: When ahead by 20%+ in Voronoi calculation

---

## Bug Fix Applied

**Previous Issue**: 
```python
if not valid_moves:
    return 'UP'  # Could crash into wall!
```

**Fixed**:
```python
if not valid_moves:
    # Try all directions, pick first that's in bounds
    for direction in [UP, DOWN, LEFT, RIGHT]:
        if in_bounds(new_position):
            return direction
    return 'RIGHT'  # Last resort
```

---

## Algorithm Selection Flow

```
1. Get valid moves (boundary + obstacle check)
   ↓
2. If no valid moves → Emergency fallback
   ↓
3. If only 1 valid move → Take it
   ↓
4. Check game state:
   - Is critical? → Use MCTS
   - Not stationary? → Use Alpha-Beta Minimax
   - Stationary? → Use Wall-Following
   ↓
5. Validate chosen move is in valid_moves
   ↓
6. Check if should use boost (safety checks)
   ↓
7. Return move (with optional :BOOST)
```

---

## Summary

**Crash Avoidance Layers**:
1. ✅ **Boundary checking** in get_valid_moves()
2. ✅ **Lookahead** in Minimax (2 moves ahead)
3. ✅ **Simulation** in MCTS (150 random playouts)
4. ✅ **Space maximization** in wall-following
5. ✅ **Boost safety** checks (distance + free space)
6. ✅ **Emergency fallback** when no valid moves

**Strengths**:
- Multi-algorithm approach adapts to game phase
- Strong lookahead prevents traps
- Boost logic is conservative and safe

**Potential Improvements**:
- Increase minimax depth to 3-4 moves
- Add flood-fill for better territory estimation
- Implement opponent move prediction
- Add endgame-specific strategies
