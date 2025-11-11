# Why Agent 2 (RevRage) is Winning

## Key Differences Between Agents

### Agent 2 (RevRage) - The Winner
**Strategy**: Legacy TronBot - Battle-tested competitive Tron AI

**Strengths:**
1. **Iterative Deepening Alpha-Beta**
   - Starts at depth 1 and increases up to depth 100
   - Uses time-based cutoff (750ms per move)
   - Keeps best move from previous depth if timeout occurs
   - More adaptive to time constraints

2. **Advanced Territory Evaluation**
   - Uses Tarjan's articulation points algorithm
   - Calculates "max articulated space" - finds largest connected territory
   - Sophisticated edge counting (K2 = 194 weight)
   - Territory scoring: K1 * (front + fillable) + K2 * edges

3. **Space-Filling Algorithm**
   - Dedicated algorithm for separated chambers
   - Iteratively searches for maximum fillable space
   - Better at endgame when agents are separated

4. **Killer Move Heuristic**
   - Stores and reuses good moves from previous searches
   - Improves move ordering in alpha-beta search
   - Reduces search time significantly

5. **Component Analysis**
   - Sophisticated connected component detection
   - Tracks red/black squares (checkerboard pattern)
   - Calculates fillable area based on color parity

### Agent 1 (Noob Agent) - Your Agent
**Strategy**: Multi-algorithm approach with MCTS, Minimax, and heuristics

**Strengths:**
1. **Multiple Algorithms**
   - Alpha-Beta Minimax (depth 2)
   - MCTS (150 simulations)
   - Wall-following heuristic
   - Chamber detection

2. **Strategic Boost Usage**
   - Conservative boost logic
   - Territory-aware boost decisions

**Weaknesses:**
1. **Fixed Depth Search**
   - Minimax only looks 2 moves ahead
   - Agent 2 can look 10-20+ moves ahead with iterative deepening
   - Misses long-term traps and opportunities

2. **Simple Evaluation**
   - Basic Voronoi territory count (my_area - opp_area)
   - Doesn't consider edge quality or articulation points
   - No sophisticated territory analysis

3. **MCTS Overhead**
   - 150 simulations may not be enough
   - Random playouts don't capture strategic play
   - Used in critical situations but may be too late

4. **Strategy Switching**
   - Switches between algorithms based on game phase
   - May not always pick the best algorithm for the situation
   - No iterative deepening to maximize thinking time

## Specific Issues

### 1. Depth Problem
```python
# Agent 1: Fixed depth
self.minimax = AlphaBetaMinimax(self.grid_width, self.grid_height, max_depth=2)

# Agent 2: Iterative deepening
for depth in range(DEPTH_INITIAL, DEPTH_MAX):  # 1 to 100
    value, line = self.alphabeta(...)
```
**Impact**: Agent 2 sees 5-10x further ahead, avoiding traps Agent 1 walks into.

### 2. Evaluation Function
```python
# Agent 1: Simple
return my_area - opp_area

# Agent 2: Sophisticated
score = K1 * (front + fillable) + K2 * edges
# K1=55, K2=194 - heavily weights edge control
```
**Impact**: Agent 2 values positions that control more edges and articulation points.

### 3. Time Management
- Agent 1: Fixed depth regardless of time available
- Agent 2: Uses all available time (750ms) to search deeper
**Impact**: Agent 2 makes better use of thinking time.

## Recommendations to Improve Agent 1

### High Priority:
1. **Implement Iterative Deepening**
   - Start at depth 1, increase until timeout
   - Keep best move from completed depth
   - Will dramatically improve play

2. **Improve Evaluation Function**
   - Add edge counting
   - Consider articulation points
   - Weight territory quality, not just quantity

3. **Increase Base Depth**
   - Change from depth 2 to at least depth 4
   - Or implement iterative deepening

### Medium Priority:
4. **Better MCTS Integration**
   - Increase simulations to 500-1000
   - Use MCTS earlier, not just in critical situations
   - Or remove MCTS and focus on deeper minimax

5. **Simplify Strategy**
   - Remove strategy switching
   - Use one strong algorithm (iterative deepening minimax)
   - Consistency beats complexity

### Low Priority:
6. **Add Killer Move Heuristic**
   - Store good moves from previous searches
   - Improves move ordering

7. **Better Chamber Detection**
   - Use Tarjan's algorithm like Agent 2
   - More accurate territory calculation

## Quick Win: Increase Minimax Depth

The easiest improvement is to change:
```python
self.minimax = AlphaBetaMinimax(self.grid_width, self.grid_height, max_depth=2)
```
to:
```python
self.minimax = AlphaBetaMinimax(self.grid_width, self.grid_height, max_depth=4)
```

This alone should significantly improve performance, though it will be slower.

## The Real Solution: Iterative Deepening

Agent 2 wins because it uses **all available thinking time** to search as deep as possible, while Agent 1 stops at depth 2 regardless of how much time is left.

Implementing iterative deepening would be the single biggest improvement to make Agent 1 competitive.
