"""
Script to run multiple games and calculate win percentage
"""
import subprocess
import time
import re
import signal
import os

def run_game():
    """Run a single game and return the result"""
    try:
        result = subprocess.run(
            ["bash", "-c", "source venv/bin/activate && python judge_engine.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout
        
        # Parse the result
        if "Winner: Agent 1" in output:
            return "WIN"
        elif "Winner: Agent 2" in output:
            return "LOSS"
        elif "draw" in output.lower():
            return "DRAW"
        else:
            return "ERROR"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        print(f"Error running game: {e}")
        return "ERROR"

def main():
    num_games = 20
    results = {"WIN": 0, "LOSS": 0, "DRAW": 0, "ERROR": 0, "TIMEOUT": 0}
    
    print(f"Running {num_games} games...")
    print("=" * 60)
    
    for i in range(1, num_games + 1):
        print(f"\nGame {i}/{num_games}:")
        result = run_game()
        results[result] += 1
        print(f"Result: {result}")
        
        # Small delay between games
        if i < num_games:
            time.sleep(2)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)
    print(f"Wins:     {results['WIN']}")
    print(f"Losses:   {results['LOSS']}")
    print(f"Draws:    {results['DRAW']}")
    print(f"Errors:   {results['ERROR']}")
    print(f"Timeouts: {results['TIMEOUT']}")
    print("-" * 60)
    
    total_completed = results['WIN'] + results['LOSS'] + results['DRAW']
    if total_completed > 0:
        win_rate = (results['WIN'] / total_completed) * 100
        print(f"Win Rate: {win_rate:.1f}% ({results['WIN']}/{total_completed} games)")
        
        if results['DRAW'] > 0:
            win_draw_rate = ((results['WIN'] + results['DRAW']) / total_completed) * 100
            print(f"Win+Draw Rate: {win_draw_rate:.1f}%")
    else:
        print("No games completed successfully")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
