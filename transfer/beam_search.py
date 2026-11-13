import argparse
import csv
import ctypes
import fcntl
import os
import pickle
import random
import shutil
import signal
import socket
import struct
import tempfile
import threading
import time
import numpy as np
import torch
import gym_compat as gc
import evaluation_framework as ef
import test_solution as ts
from analysis import plan_metrics as pm
from transfer import env as et

"""
Beam search with the trained model on a city: the model is not changed, it only guides the search.

Every live beam member is a forked process holding a full environment. In each step every member tries the
top-k actions of the model (each in its own forked child); of all resulting plans the B best are kept, ranked by

    key = best score so far + value_weight * max_a Q(next observation, a)

i.e. the score already reached plus the model's estimate of the improvement still to come (the reward is the gain
over the best score so far, so Q estimates exactly that). Plans with the same set of stations are kept once.

Greedy backbone: one of the B places always holds the greedy episode of the model (top-1 action from the start), so
the search covers the zero-shot result and adds B - 1 places for alternatives. Plans that finish (budget used up)
are kept aside; the result is the best plan over all finished and live members. After the backbone has finished, the
search stops when the best score has not improved for --patience steps.

Service metrics (definitions of test_solution.py): the average travel and waiting time are printed for the existing
plan at the start, every --log-every steps for the current plan of the beam member with the best score, and at the
end for the saved plan (--save-plan, same numbers as analysis/plan_metrics.py). They are only read from the members,
so the search and its result do not change; --log-every 0 turns the periodic lines off.

With --beam 1 --topk 1 this is exactly the greedy episode of transfer/evaluate.py (same random draws).

Memory protection: all beam members run in their own process group. A watchdog thread in the coordinator checks
every few seconds (a) the memory of the coordinator plus all beam members (PSS: copy-on-write pages shared after a
fork are counted once) against --max-mem-gb and (b) the free memory of the whole machine (MemAvailable) against
--min-available-gb. If either limit is crossed, the whole process group is killed with SIGKILL and the run stops
with exit code 3, instead of the machine running out of memory and hanging (no ssh, hard reset needed).

    python -m transfer.beam_search --city Taipei --model runs/source_Tainan/best_source.zip --beam 4 --topk 2 --eval-seed 1
"""

GB_IN_KB = 1024 ** 2

FIELDS = ["city", "label", "model", "beam", "topk", "value_weight", "eval_seed", "existing_score", "score",
          "improvement", "n_stations", "episode_steps", "expansions", "seconds", "action_counts"]


def send(conn, obj):
    data = pickle.dumps(obj)
    conn.sendall(struct.pack("Q", len(data)) + data)


def recv(conn):
    def read(n):
        chunks = b""
        while len(chunks) < n:
            chunk = conn.recv(n - len(chunks))
            if not chunk:
                raise EOFError("connection closed")
            chunks += chunk
        return chunks
    return pickle.loads(read(struct.unpack("Q", read(8))[0]))


def fork_keep_random():
    """
    os.fork(), but the child keeps the state of the random module. Python reseeds `random` from the OS in every
    forked child (os.register_at_fork), and env_plus draws a random node with random.choice whenever every station is
    full (actions 2/3) or nothing can be moved (action 4); without this a beam member would not follow the seeded
    episode, so results were not reproducible and B = k = 1 differed from transfer/evaluate.py.
    """
    state = random.getstate()
    pid = os.fork()
    if pid == 0:
        random.setstate(state)
    return pid


class ScoredStationPlacement(et.TransferStationPlacement):
    """also remembers the score of the current plan (last_score), not only the best one"""

    def evaluation(self):
        reward = 0
        self.prepare_score()
        new_score, _, _, _, _, _ = ef.norm_score(self.plan_instance.plan, self.node_list,
                                                 self.plan_instance.norm_benefit, self.plan_instance.norm_charg,
                                                 self.plan_instance.norm_wait, self.plan_instance.norm_travel)
        new_score = max(new_score, -25)
        self.last_score = new_score
        if new_score - self.best_score > 0:
            reward += (new_score - self.best_score)
            self.best_score = new_score
            self.best_plan = self.plan_instance.plan.copy()
            self.best_node_list = self.node_list.copy()
        return reward


def plan_signature(plan):
    return hash(tuple(sorted((station[0][0], tuple(station[1])) for station in plan)))


def q_values(model, obs):
    with torch.no_grad():
        return model.q_net(torch.as_tensor(obs[None], dtype=torch.float32)).numpy()[0].astype(float)


class Worker:
    """code running in the beam member processes"""

    def __init__(self, env, model, socket_path):
        self.env, self.model, self.socket_path = env, model, socket_path

    def report(self, my_id, parent, action, obs, done, error=None):
        env = self.env
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(self.socket_path)
        send(conn, dict(id=my_id, parent=parent, action=action, done=done, error=error,
                        best=env.best_score, last=getattr(env, "last_score", env.best_score),
                        budget=env.budget, n_stations=len(getattr(env, "best_plan", env.plan_instance.plan)),
                        sig=plan_signature(env.plan_instance.plan),
                        q=None if done or error else q_values(self.model, obs)))
        return conn

    def serve(self, conn, my_id):
        """command loop of one beam member; a forked child continues in this same loop as the new member"""
        while True:
            command = recv(conn)
            if command[0] == "expand":
                for action, child_id in command[1]:
                    if fork_keep_random() == 0:
                        conn.close()
                        try:
                            obs, _, done, _ = gc.step(self.env, action)
                            error = None
                        except Exception as exc:  # report and stop this branch
                            obs, done, error = None, True, repr(exc)
                        conn, my_id = self.report(child_id, my_id, action, obs, done, error), child_id
                        break
                else:
                    os._exit(0)  # parent: all expansions live on in the children
            elif command[0] == "metrics":  # read only: the member goes on unchanged
                send(conn, ts.service_progress(command[1], self.env.node_list, self.env.plan_instance.plan))
            elif command[0] == "save":
                os.makedirs(command[1], exist_ok=True)
                pickle.dump(self.env.best_plan, open(os.path.join(command[1], "plan_RL.pkl"), "wb"))
                with open(os.path.join(command[1], "nodes_RL.txt"), "w") as file:
                    file.write(str(self.env.best_node_list))
                send(conn, "saved")
            else:  # die
                os._exit(0)


class Coordinator:
    def __init__(self, listener):
        self.listener, self.conns, self.next_id = listener, {}, 1

    def collect(self, n):
        reports = []
        for _ in range(n):
            conn, _ = self.listener.accept()
            report = recv(conn)
            self.conns[report["id"]] = conn
            reports.append(report)
        return reports

    def command(self, my_id, *command):
        send(self.conns[my_id], command)

    def service(self, my_id, step):
        """progress line with the average travel / waiting time of the member's current plan"""
        self.command(my_id, "metrics", step)
        return recv(self.conns[my_id])

    def kill(self, my_id):
        conn = self.conns.pop(my_id)
        try:
            send(conn, ("die",))
        except OSError:
            pass
        conn.close()

    def expand(self, beam, topk, backbone):
        """
        send every beam member its top-k actions; the member processes are replaced by their children.
        Returns the number of children and the id of the backbone's top-1 child (None without backbone).
        """
        n, next_backbone = 0, None
        for state in beam:
            actions = np.argsort(-state["q"], kind="stable")[:topk]
            jobs = []
            for action in actions:
                jobs.append((int(action), self.next_id))
                self.next_id += 1
            if state["id"] == backbone:
                next_backbone = jobs[0][1]
            self.command(state["id"], "expand", jobs)
            self.conns.pop(state["id"]).close()
            n += len(jobs)
        return n, next_backbone


def mem_available_gb():
    """memory the whole machine can still give out without swapping (GB)"""
    with open("/proc/meminfo") as file:
        for line in file:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / GB_IN_KB
    return float("inf")


def group_pss_gb(pgid, extra_pids=()):
    """PSS of all processes in process group pgid plus extra_pids (GB); shared copy-on-write pages count once"""
    total = 0
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            if pid not in extra_pids:
                with open("/proc/{}/stat".format(pid)) as file:
                    # "pid (comm) state ppid pgrp ..."; comm may contain spaces and brackets
                    if int(file.read().rsplit(")", 1)[1].split()[2]) != pgid:
                        continue
            with open("/proc/{}/smaps_rollup".format(pid)) as file:
                for line in file:
                    if line.startswith("Pss:"):
                        total += int(line.split()[1])
                        break
        except (OSError, ValueError, IndexError):  # process ended meanwhile or is not ours
            continue
    return total / GB_IN_KB


class MemoryGuard(threading.Thread):
    """kills the beam (process group pgid) and ends the run before the machine runs out of memory"""

    def __init__(self, pgid, max_gb, min_available_gb, socket_path, interval=2.0):
        super().__init__(daemon=True)
        self.pgid, self.max_gb, self.min_available_gb = pgid, max_gb, min_available_gb
        self.socket_path, self.interval, self.peak_gb = socket_path, interval, 0.0

    def run(self):
        while True:
            time.sleep(self.interval)
            reason = None
            available = mem_available_gb()
            if self.min_available_gb > 0 and available < self.min_available_gb:
                reason = "machine has only {:.1f} GB available (--min-available-gb {:g})".format(
                    available, self.min_available_gb)
            used = group_pss_gb(self.pgid, (os.getpid(),))
            self.peak_gb = max(self.peak_gb, used)
            if reason is None and self.max_gb > 0 and used > self.max_gb:
                reason = "beam processes use {:.1f} GB (--max-mem-gb {:g})".format(used, self.max_gb)
            if reason:
                try:
                    os.killpg(self.pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                print("MEMORY GUARD: {}; all beam members killed, search aborted".format(reason), flush=True)
                shutil.rmtree(os.path.dirname(self.socket_path), ignore_errors=True)
                os._exit(3)


def reap():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--model", default="runs/source_Tainan/best_source.zip")
    parser.add_argument("--beam", type=int, default=4, help="beam width B")
    parser.add_argument("--topk", type=int, default=2, help="actions tried per beam member (by Q-value)")
    parser.add_argument("--value-weight", type=float, default=1.0, help="weight of max Q in the ranking key")
    parser.add_argument("--eval-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=200,
                        help="after the greedy backbone has finished: stop when the best score has not improved "
                             "for this many steps")
    parser.add_argument("--label", default="beam")
    parser.add_argument("--csv", default="Results/Transfer/beam.csv")
    parser.add_argument("--save-plan", default=None)
    parser.add_argument("--log-every", type=int, default=25,
                        help="progress line with score and average travel / waiting time every n steps (0 = off)")
    parser.add_argument("--max-mem-gb", type=float, default=16.0,
                        help="abort when coordinator + beam members use more than this many GB (PSS); 0 = off")
    parser.add_argument("--min-available-gb", type=float, default=4.0,
                        help="abort when the machine has less than this many GB available; 0 = off")
    args = parser.parse_args()

    torch.set_num_threads(1)  # many forked processes, one step each
    ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER: orphaned beam members are reaped here
    repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # transfer/..
    graph_dir = os.path.join(repository_root, "Graph", args.city)
    env = ScoredStationPlacement(os.path.join(graph_dir, args.city + ".graphml"),
                                    os.path.join(graph_dir, "node_list_" + args.city + ".txt"),
                                    os.path.join(graph_dir, "existingplan_" + args.city + ".pkl"))
    model = gc.load_dqn(args.model, env)
    model.q_net.eval()

    socket_dir = tempfile.mkdtemp(prefix="pcrl_beam_")  # private directory (mode 0700), removed at the end
    socket_path = os.path.join(socket_dir, "coordinator.sock")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path)
    listener.listen(1024)
    listener.settimeout(1800)
    coordinator = Coordinator(listener)
    worker = Worker(env, model, socket_path)

    start = time.time()
    random.seed(args.eval_seed)  # same seeding as transfer.evaluate.run_episode
    np.random.seed(args.eval_seed)
    obs = gc.reset(env)
    existing_score = env.best_score
    first_member = fork_keep_random()
    if first_member == 0:
        os.setpgid(0, 0)  # own process group; every later beam member inherits it
        listener.close()
        worker.serve(worker.report(0, None, None, obs, False), 0)
    try:  # set in both processes, so the group exists whichever runs first
        os.setpgid(first_member, first_member)
    except OSError:
        pass
    # the thread is started only after the fork: beam members are forked from member processes, never from here
    guard = MemoryGuard(first_member, args.max_mem_gb, args.min_available_gb, socket_path)
    guard.start()
    try:
        beam = coordinator.collect(1)
        lineage = {0: (None, None)}
        best_done, steps, expansions = None, 0, 0
        backbone, best_overall, last_improvement = 0, existing_score, 0
        print("start: score {:.4f}, Q {}".format(existing_score, np.round(beam[0]["q"], 3)), flush=True)
        print("existing plan, " + coordinator.service(0, 0), flush=True)
        while beam and steps < args.max_steps:
            if backbone is None and steps - last_improvement >= args.patience:
                print("step {}: no improvement for {} steps, search stopped".format(steps, args.patience), flush=True)
                break
            n, backbone = coordinator.expand(beam, args.topk, backbone)
            reports = coordinator.collect(n)
            reap()
            steps += 1
            expansions += n
            alive, backbone_alive = [], None
            for report in reports:
                lineage[report["id"]] = (report["parent"], report["action"])
                if report["best"] > best_overall:
                    best_overall, last_improvement = report["best"], steps
                if report["id"] == backbone and (report["error"] or report["done"]):
                    backbone = None  # the greedy episode has ended
                if report["error"]:
                    print("branch {} failed: {}".format(report["id"], report["error"]), flush=True)
                    coordinator.kill(report["id"])
                elif report["done"]:
                    if best_done is None or report["best"] > best_done["best"]:
                        if best_done is not None:
                            coordinator.kill(best_done["id"])
                        best_done = report
                    else:
                        coordinator.kill(report["id"])
                else:
                    report["key"] = report["best"] + args.value_weight * float(np.max(report["q"]))
                    if report["id"] == backbone:
                        backbone_alive = report
                    else:
                        alive.append(report)
            alive.sort(key=lambda r: (r["key"], r["last"]), reverse=True)
            beam, seen = [], set()
            if backbone_alive is not None:  # the greedy episode always keeps its place
                beam.append(backbone_alive)
                seen.add(backbone_alive["sig"])
            for report in alive:
                if len(beam) < args.beam and report["sig"] not in seen:
                    beam.append(report)
                    seen.add(report["sig"])
                else:
                    coordinator.kill(report["id"])
            if (args.log_every > 0 and steps % args.log_every == 0) or not beam:
                top = max(beam, key=lambda r: r["key"]) if beam else None
                print("step {} beam {} best live {:.4f} (key {:.4f}, {} stations, budget {:.0f}) best done {} "
                      "({:.0f}s){}".format(steps, len(beam), max([r["best"] for r in beam], default=float("nan")),
                                           top["key"] if top else float("nan"), top["n_stations"] if top else 0,
                                           top["budget"] if top else 0,
                                           "{:.4f}".format(best_done["best"]) if best_done else "-",
                                           time.time() - start, "" if backbone is None else " backbone live"),
                      flush=True)
                if beam:
                    leader = max(beam, key=lambda r: r["best"])
                    print("  current plan of the best member, " + coordinator.service(leader["id"], steps), flush=True)

        candidates = beam + ([best_done] if best_done else [])
        winner = max(candidates, key=lambda r: r["best"])
        if args.save_plan:
            coordinator.command(winner["id"], "save", os.path.abspath(args.save_plan))
            assert recv(coordinator.conns[winner["id"]]) == "saved"
        for my_id in list(coordinator.conns):
            coordinator.kill(my_id)

        path, node = [], winner["id"]
        while lineage[node][0] is not None:
            path.append(lineage[node][1])
            node = lineage[node][0]
        counts = np.bincount(path, minlength=5).tolist()
        row = dict(city=args.city, label=args.label, model=args.model, beam=args.beam, topk=args.topk,
                   value_weight=args.value_weight, eval_seed=args.eval_seed, existing_score=existing_score,
                   score=winner["best"], improvement=winner["best"] - existing_score,
                   n_stations=winner["n_stations"], episode_steps=len(path), expansions=expansions,
                   seconds=round(time.time() - start, 1), action_counts=" ".join(map(str, counts)))
        print(row, flush=True)
        print("peak memory of coordinator + beam members: {:.1f} GB (PSS)".format(guard.peak_gb), flush=True)
        if args.save_plan:  # all members are gone, so the caches of this process can be used
            final_nodes, final_plan = pm.load_plan(args.city, args.save_plan)
            print("saved plan {}, ".format(args.save_plan) + ts.service_progress(len(path), final_nodes, final_plan)
                  .split(": ", 1)[1], flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "a", newline="") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
    finally:
        for my_id in list(coordinator.conns):
            coordinator.kill(my_id)
        try:  # also members not connected right now (e.g. in the middle of a step after Ctrl-C or an exception)
            os.killpg(first_member, signal.SIGKILL)
        except ProcessLookupError:
            pass
        listener.close()
        shutil.rmtree(socket_dir, ignore_errors=True)
        reap()


if __name__ == '__main__':
    main()
