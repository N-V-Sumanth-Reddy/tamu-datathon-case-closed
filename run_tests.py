#!/usr/bin/env python3
"""Run multiple games and calculate win statistics."""

import subprocess
import re
import time
from collections import defaultdict

def run_single_game(game_num):
    """Run a single game and return the winner."""
    print(f"\n{'='*60}")
    print(f"Running Game {game_num}/20...")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            ['python', 'judge_engine.py'],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per game
        )
        
        output = result.stdout
        
        # Extract winner from output
        if "Winner: Agent 1" in output:
            winner = "Agent 1 (Noob Agent)"
        elif "Winner: Agent 2" in output:
            winner = "Agent 2 (RevRage)"
        elif "Draw" in output or "draw" in output:
            winner = "Draw"
        else:
            winner = "Unknown"
        
        # Extract turn count
        turn_match = re.search(r'=== Turn (\d+) ===', output)
        turns = int(turn_match.group(1)) if turn_match else 0
        
        # Extract final trail lengths
        trail_match = re.search(
            r'Agent 1: Trail Length=(\d+).*\n.*Agent 2: Trail Length=(\d+)',
            output
        )
        if trail_match:
            agent1_length = int(trail_match.group(1))
            agent2_length = int(trail_match.group(2))
        else:
            agent1_length = 0
            agent2_length = 0
        
        print(f"✓ Game {game_num} complete: {winner}")
        print(f"  Turns: {turns}, Agent1: {agent1_length}, Agent2: {agent2_length}")
        
        return {
            'game': game_num,
            'winner': winner,
            'turns': turns,
            'agent1_length': agent1_length,
            'agent2_length': agent2_length
        }
        
    except subprocess.TimeoutExpired:
        print(f"✗ Game {game_num} timed out")
        return {
            'game': game_num,
            'winner': 'Timeout',
            'turns': 0,
            'agent1_length': 0,
            'agent2_length': 0
        }
    except Exception as e:
        print(f"✗ Game {game_num} error: {e}")
        return {
            'game': game_num,
            'winner': 'Error',
            'turns': 0,
            'agent1_length': 0,
            'agent2_length': 0
        }

def main():
    print("="*60)
    print("TRON AI TOURNAMENT - 20 GAMES")
    print("Agent 1: Noob Agent (ParticipantX)")
    print("Agent 2: RevRage (TowerDuo)")
    print("="*60)
    
    results = []
    stats = defaultdict(int)
    
    start_time = time.time()
    
    for i in range(1, 21):
        result = run_single_game(i)
        results.append(result)
        stats[result['winner']] += 1
        
        # Brief pause between games
        time.sleep(1)
    
    elapsed_time = time.time() - start_time
    
    # Calculate statistics
    total_games = len(results)
    agent1_wins = stats.get("Agent 1 (Noob Agent)", 0)
    agent2_wins = stats.get("Agent 2 (RevRage)", 0)
    draws = stats.get("Draw", 0)
    errors = stats.get("Error", 0) + stats.get("Timeout", 0) + stats.get("Unknown", 0)
    
    valid_games = total_games - errors
    
    # Calculate averages
    total_turns = sum(r['turns'] for r in results if r['turns'] > 0)
    avg_turns = total_turns / valid_games if valid_games > 0 else 0
    
    total_agent1_length = sum(r['agent1_length'] for r in results)
    total_agent2_length = sum(r['agent2_length'] for r in results)
    avg_agent1_length = total_agent1_length / valid_games if valid_games > 0 else 0
    avg_agent2_length = total_agent2_length / valid_games if valid_games > 0 else 0
    
    # Print results
    print("\n" + "="*60)
    print("TOURNAMENT RESULTS")
    print("="*60)
    print(f"\nTotal Games: {total_games}")
    print(f"Valid Games: {valid_games}")
    print(f"Errors/Timeouts: {errors}")
    print(f"\nTime Elapsed: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Average Game Duration: {elapsed_time/total_games:.1f} seconds")
    
    print("\n" + "-"*60)
    print("WIN STATISTICS")
    print("-"*60)
    print(f"Agent 1 (Noob Agent) Wins:  {agent1_wins:2d} ({agent1_wins/valid_games*100:5.1f}%)")
    print(f"Agent 2 (RevRage) Wins:     {agent2_wins:2d} ({agent2_wins/valid_games*100:5.1f}%)")
    print(f"Draws:                      {draws:2d} ({draws/valid_games*100:5.1f}%)")
    
    print("\n" + "-"*60)
    print("GAME STATISTICS")
    print("-"*60)
    print(f"Average Turns per Game:     {avg_turns:.1f}")
    print(f"Average Agent 1 Trail:      {avg_agent1_length:.1f}")
    print(f"Average Agent 2 Trail:      {avg_agent2_length:.1f}")
    
    # Detailed results
    print("\n" + "-"*60)
    print("DETAILED RESULTS")
    print("-"*60)
    print(f"{'Game':<6} {'Winner':<25} {'Turns':<7} {'A1 Trail':<10} {'A2 Trail':<10}")
    print("-"*60)
    for r in results:
        winner_short = r['winner'].split('(')[0].strip() if '(' in r['winner'] else r['winner']
        print(f"{r['game']:<6} {winner_short:<25} {r['turns']:<7} {r['agent1_length']:<10} {r['agent2_length']:<10}")
    
    print("\n" + "="*60)
    if agent1_wins > agent2_wins:
        print(f"🏆 CHAMPION: Agent 1 (Noob Agent) with {agent1_wins} wins!")
    elif agent2_wins > agent1_wins:
        print(f"🏆 CHAMPION: Agent 2 (RevRage) with {agent2_wins} wins!")
    else:
        print(f"🤝 TIE: Both agents won {agent1_wins} games each!")
    print("="*60)

if __name__ == "__main__":
    main()
