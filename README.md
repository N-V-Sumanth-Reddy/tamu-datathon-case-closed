# TAMU Datathon - Case Closed Challenge

## 🏆 Tron AI Agent Competition

An advanced Tron AI agent built for the TAMU Datathon Case Closed Challenge, featuring multiple sophisticated algorithms and a journey through classical AI and reinforcement learning.

---

## 🎮 About the Challenge

The Case Closed Challenge is a Tron-style game where two AI agents compete on a grid:
- Agents leave permanent trails behind them
- Crashing into trails, boundaries, or each other means elimination
- **Torus board**: Wraparound mechanics (edges connect)
- **Boost mechanic**: 3 boosts per game to jump 2 spaces
- Last agent standing wins

**Grid Size**: 20x18  
**Max Turns**: 500  
**Time per Move**: 4 seconds (with 2 retry attempts)

---

## 🤖 Agent Architecture

### The Beast - Classical AI Approach

My agent uses a multi-algorithm approach inspired by **Google AI Challenge winners**:

#### 1. **Alpha-Beta Minimax with Iterative Deepening** ⭐
- Searches up to depth 20 moves ahead
- Uses all available thinking time (0.7s per move)
- Killer move heuristic for optimal pruning
- Advanced evaluation function:
  - Territory control: 50 points per square
  - Edge mobility: 10 points per edge
  - Articulation risk: -20 points per bottleneck

#### 2. **Chamber Detection Module**
- Tarjan's articulation points algorithm
- Voronoi territory computation
- Detects when agents are separated
- Switches strategies based on game phase

#### 3. **Monte Carlo Tree Search (MCTS)**
- 500 simulations for critical decisions
- 20-step random playouts
- Triggered when:
  - Distance to opponent ≤ 4
  - Only 2 or fewer valid moves
  - Board >70% filled

#### 4. **Strategic Boost System**
- Conservative boost usage with safety checks
- Territory-aware boost decisions
- Collision avoidance
- Timing optimization (early/mid/late game)

#### 5. **Torus Wraparound Support**
- Full wraparound mechanics
- All algorithms support board edges
- No corner traps

---

## 📊 The Journey: From RL Dreams to Classical Reality

### Phase 1: Building The Beast (Classical AI)
I started by implementing sophisticated classical algorithms:
- ✅ Iterative deepening minimax
- ✅ Advanced evaluation functions
- ✅ MCTS for tactical decisions
- ✅ Chamber detection

**Result**: Strong agent that could think 15+ moves ahead.

### Phase 2: The RL Experiment (A2C)
Inspired by **AlphaGo's training strategy**, I built an RL agent:

**Training Pipeline**:
1. Random move exploration
2. Training against classical baselines
3. Training against The Beast
4. Self-play for mastery

**The Problem**: Training curve clamped to 0 when facing The Beast. The agent couldn't learn—it just kept losing.

### Phase 3: The Pivot (PPO)
Switched to **PPO (Proximal Policy Optimization)**:
- A2C + clipping to handle difficult training scenarios
- Better gradient stability

**Result**: 35% win rate against The Beast! But time was running out...

### Phase 4: The Pragmatic Choice
With the deadline approaching, I enhanced The Beast instead of continuing with RL.

**Confidence Level**: High. I assumed others would struggle with RL in the short timeframe.

---

## 💔 The Competition: The Lesson

### The Defeat
We lost.

### The Investigation
I immediately checked the winning code. My heart sank.

**Their approach**: Nearly identical to mine.
- Same algorithms
- Same Google AI Challenge inspiration
- No RL
- Just classical AI

### The Revelation
I asked AI to analyze the difference. One word:

**BOOSTING**

I went back to the problem statement. There it was, in the optional features:

> Agents can use boost moves (UP:BOOST, DOWN:BOOST, etc.) for a limited number of times (3). The agent jumps 2 spaces in one move.

My eyes filled with tears.

I had implemented boost logic. But not strategically enough. Not aggressively enough. The winners had **mastered boost timing**.

### The Validation
I enhanced my boost strategy to match the winning approach.

I ran tests against the winning code.

**My enhanced agent won.**

😢

---

## 📚 Lessons Learned

### 1. **Read Every Word of the Specification**
The difference between victory and defeat wasn't algorithm sophistication—it was understanding ALL the game mechanics.

### 2. **Simple Features Can Be Game-Changers**
Boost moves seemed like a minor feature. They were actually the most powerful tool in the game.

### 3. **Classical AI Still Competes**
In short timeframes, well-implemented classical algorithms can match or beat RL approaches.

### 4. **Failure Is the Best Teacher**
This loss taught me more than any victory could have.

---

## 🚀 The Journey Continues

### Will I Stop Here?

**No.**

While writing this, there's a training simulation running in the background.

### The Next Chapter: Ultimate RL Agent

I'm building the **ultimate RL agent** with:
- ✅ Proper boost integration from day one
- ✅ Multiple RL algorithms (A2C, PPO, SAC, DQN)
- ✅ Curriculum learning strategies
- ✅ Self-play with population-based training
- ✅ Lessons learned from this competition

**Goal**: Build an RL agent that beats The Beast (now with proper boost strategy).

---

## 🛠️ Technical Stack

**Languages**: Python 3.13  
**Frameworks**: Flask (API), NumPy (computation)  
**Algorithms**: Alpha-Beta, MCTS, Iterative Deepening, Voronoi, Tarjan's  
**RL (Experimental)**: A2C, PPO, PyTorch  

---

## 📁 Repository Structure

```
├── agent.py                    # The Enhanced Beast (Main Agent)
├── sample_agent.py             # Reference Agent (TronBot)
├── judge_engine.py             # Game Judge/Referee
├── case_closed_game.py         # Game Mechanics
├── local-tester.py             # API Compliance Tester
├── run_tests.py                # Tournament Runner
├── requirements.txt            # Dependencies
│
├── JOURNEY.md                  # This file - The full story
├── ENHANCEMENTS.md             # Technical improvements
├── AGENT_ALGORITHMS.md         # Algorithm documentation
├── AGENT_COMPARISON.md         # Agent analysis
├── TESTING_GUIDE.md            # How to run tests
│
└── rl_agent/                   # [Coming Soon] RL experiments
    ├── a2c_agent.py
    ├── ppo_agent.py
    └── training_logs/
```

---

## 🚦 Quick Start

### Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Single Game
```bash
# Terminal 1: Start Agent 1
python agent.py

# Terminal 2: Start Agent 2
python sample_agent.py

# Terminal 3: Run Judge
python judge_engine.py
```

### Test API Compliance
```bash
# Terminal 1: Start your agent
python agent.py

# Terminal 2: Run tests
python local-tester.py
```

---

## 📈 Current Status

**Classical Agent (The Beast)**: ✅ Complete
- Iterative deepening (depth 20)
- Advanced evaluation
- Strategic boost usage
- Torus wraparound support

**RL Agent**: 🔄 In Development
- A2C: Trained (35% win rate)
- PPO: Trained (35% win rate)
- Next: Enhanced training with boost mastery

---

## 🎯 Future Updates

- [ ] Complete RL agent training with boost integration
- [ ] Tournament results (RL vs Classical)
- [ ] Training curves and analysis
- [ ] Open-source RL training framework
- [ ] Video demonstrations
- [ ] Performance benchmarks

---

## 🤝 Contributing

This is a personal learning journey, but insights and suggestions are welcome!

---

## 📝 License

MIT License - Feel free to learn from and build upon this code.

---

## 🙏 Acknowledgments

- **TAMU Datathon** for organizing the challenge
- **Google AI Challenge** for algorithm inspiration
- **AlphaGo** for RL training strategy inspiration
- **The winning teams** for showing me what I missed

---

## 💭 Final Thoughts

> "I have not failed. I've just found 10,000 ways that won't work."  
> — Thomas Edison

This competition taught me that:
- **Sophistication ≠ Success**
- **Understanding > Implementation**
- **Persistence > Perfection**

The journey continues. Stay tuned for the RL comeback story.

---

**Status**: 🔥 Training in Progress  
**Next Update**: When the RL agent beats The Beast  
**Follow**: [GitHub](https://github.com/N-V-Sumanth-Reddy) for updates

---

*Built with determination, debugged with tears, enhanced with lessons learned.*
