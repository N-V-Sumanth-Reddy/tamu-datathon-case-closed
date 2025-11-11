# Testing Guide for Case Closed Agent

## Setup Complete! ✅

Your environment is now ready to test your agents. Here's what was done:

1. **Virtual Environment**: Created `.venv` with Python 3.13.7
2. **Dependencies Installed**: Flask, requests, numpy
3. **API Compliance**: Your agent passes all compliance checks!

## Running Tests

### 1. Test API Compliance (Single Agent)

In one terminal:
```bash
source .venv/bin/activate
python agent.py
```

In another terminal:
```bash
source .venv/bin/activate
python local-tester.py
```

**Result**: ✅ All tests passed!

### 2. Run Full Game (Two Agents)

You need **3 separate terminals**:

**Terminal 1** - Start your agent (port 5008):
```bash
source .venv/bin/activate
python agent.py
```

**Terminal 2** - Start sample agent (port 5009):
```bash
source .venv/bin/activate
python sample_agent.py
```

**Terminal 3** - Run the judge engine:
```bash
source .venv/bin/activate
python judge_engine.py
```

The judge will coordinate the game between both agents and display the board state after each turn.

## Current Status

- **agent.py**: Running on http://localhost:5008 ✅
- **sample_agent.py**: Running on http://localhost:5009 ✅
- **Game**: Successfully tested and running ✅

## Agent Details

- **Your Agent**: "Noob Agent" (ParticipantX)
  - Advanced Tron AI with modular strategies
  - Implements: Opponent Modeling, Chamber Detection, Alpha-Beta Pruning, MCTS
  
- **Sample Agent**: "RevRage" (TowerDuo)
  - Legacy TronBot implementation
  - Uses minimax with alpha-beta pruning and space-filling algorithms

## Tips

- The judge waits 5 seconds on startup for agents to be ready
- Each agent gets 2 attempts per move with 4-second timeout
- If both attempts fail, agents get 5 random moves before forfeiting
- Agents can use boost moves with format "DIRECTION:BOOST"
- Maximum 500 turns per game

## Next Steps

1. Modify your agent strategy in `agent.py`
2. Test with `local-tester.py` to ensure API compliance
3. Run full games against `sample_agent.py` to test performance
4. Use `run_multiple_games.py` for batch testing (if available)

Happy coding! 🚀
