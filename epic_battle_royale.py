import argparse
import sys
import os
from pong_testbench import PongTestbench
from multiprocessing import Process, Queue
from matplotlib import font_manager
from time import sleep
import importlib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=str, help="Directory with agents.")
parser.add_argument("--render", "-r", action="store_true", help="Render the competition.")
parser.add_argument("--games", "-g", type=int, default=100, help="Number of games.")
parser.add_argument("--max_proc", "-p", type=int, default=4, help="Max number of processes.")

args = parser.parse_args()


def run_test(id1, agent1_dir, id2,  agent2_dir, queue, games, render):
    sys.path.insert(0, agent1_dir)
    orig_wd = os.getcwd()
    import agent
    os.chdir(agent1_dir)
    agent1 = agent.Agent()
    agent1.load_model()
    os.chdir(orig_wd)
    del sys.path[0]

    sys.path.insert(0, agent2_dir)
    importlib.reload(agent)
    os.chdir(agent2_dir)
    agent2 = agent.Agent()
    agent2.load_model()
    os.chdir(orig_wd)
    del sys.path[0]

    testbench = PongTestbench(render)
    testbench.init_players(agent1, agent2)
    testbench.run_test(games)

    wins1, games = testbench.get_agent_score(agent1)
    wins2, games = testbench.get_agent_score(agent2)

    name1 = agent1.get_name()
    name2 = agent2.get_name()

    queue.put((id1, id2, wins1, wins2, name1, name2, games))


def get_directories(top_dir):
    subdir_list = []
    subdir_gen = os.walk(top_dir)
    for dir, subdirs, files in subdir_gen:
        if "__pycache__" in dir:
            continue
        if "agent.py" not in files:
            print("Warn: No agent.py found in %s. Skipping." % dir)
            continue
        subdir_list.append(dir)
        print("%s added to directory list." % dir)
    return subdir_list


def epic_battle_royale(top_dir, max_proc=4):
    directories = get_directories(top_dir)
    names = ["__unknown__"] * len(directories)
    procs = []
    result_queue = Queue()
    print("Finished scanning for agents; found:", len(directories))

    for i1, d1 in enumerate(directories):
        for i2, d2 in enumerate(directories):
            if i1 == i2:
                continue
            pargs = (i1, d1, i2, d2, result_queue, args.games, args.render)
            proc = Process(target=run_test, args=pargs)
            procs.append(proc)
            print("Living procs:", sum(p.is_alive() for p in procs))
            while sum(p.is_alive() for p in procs) >= max_proc:
                sleep(0.3)
            print("Starting process")
            proc.start()
            sleep(1)

    for p in procs:
        p.join()

    # Fetch all results from the queue
    no_agents = len(directories)
    games_won = np.zeros((no_agents, no_agents), dtype=np.int32)

    while result_queue.qsize() > 0:
        id1, id2, wins1, wins2, name1, name2, games = result_queue.get()
        if wins1 + wins2 != games:
            print("Woops, wins dont sum up.")
        games_won[id1, id2] += wins1
        games_won[id2, id1] += wins2
        names[id1] = name1
        names[id2] = name2

    np.save("brres", games_won)

    # Format: Wins of ROW versus COLUMN
    np.savetxt("battle_royale_results.txt", games_won, fmt="%d")
    np.savetxt("battle_royale_players.txt", directories, fmt="%s")

    # Sum across columns to get total wins of each agent
    total_wins = games_won.sum(axis=1)

    # And across rows to get total losses.
    total_losses = games_won.sum(axis=0)
    agent_wins = list(zip(total_wins, total_losses, names, directories))
    agent_wins.sort(key=lambda x: -x[0])

    resfile = open("leaderboard.txt", "w")
    print("")
    print("-"*80)
    print("--- LEADERBOARD ---")
    for i, (wins, losses, name, dir) in enumerate(agent_wins):
        line = "%d. %s with %d wins (winrate %.2f%%) (from %s)" % (i+1, name, wins, wins/(wins+losses)*100, dir)
        resfile.write(line+"\n")
        print(line)
    resfile.close()
    print("-"*80)
    print("")

    print("Finished!")


if __name__ == "__main__":
    epic_battle_royale(args.dir, args.max_proc)
