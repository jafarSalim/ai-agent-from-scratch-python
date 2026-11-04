"""Measures how much each guard helps as the model makes more mistakes."""
import json, math, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agent import *
from benchmark_tasks import *
from llm import FaultyPlanner

OUT = pathlib.Path("results")
OUT.mkdir(exist_ok=True)
SETS = {"No guards": Guards(False, False, False), "Show errors only": Guards(False, True, False),
        "Repair JSON only": Guards(True, False, False), "All guards": Guards(True, True, True)}
FAULTS, SEEDS = [0, .05, .1, .2, .3, .4, .5], 40


def good(res, t):
    try:
        return abs(float(res.answer) - t["truth"]) <= 1e-6 * abs(t["truth"])
    except (TypeError, ValueError):
        return False


table, steps = {}, {}
for name, g in SETS.items():
    table[name], steps[name] = [], []
    for f in FAULTS:
        ok, st = [], []
        for t in TASKS:
            for s in range(SEEDS):
                r = run_agent(t["q"], FaultyPlanner(t, f, s), g)
                ok.append(good(r, t))
                st.append(r.steps)
        table[name].append(float(np.mean(ok)))
        steps[name].append(float(np.mean(st)))
    print(name, [round(100 * x, 1) for x in table[name]], "steps", round(steps[name][-1], 1))
FAMILIES = {"Wavelength": TASKS[:8], "Thermal noise": TASKS[8:14], "Shannon capacity": TASKS[14:20], "Clock cycles": TASKS[20:]}


def outcome(r, t):
    if good(r, t):
        return "Correct"
    if r.stopped == "step limit":
        return "Hit step limit"
    if r.stopped:
        return "Crashed"
    return "Wrong answer"


LABELS = ["Correct", "Wrong answer", "Crashed", "Hit step limit"]
family = {k: [float(np.mean([good(run_agent(t["q"], FaultyPlanner(t, f, s), SETS["All guards"]), t) for t in v for s in range(SEEDS)]))
              for f in FAULTS] for k, v in FAMILIES.items()}
reasons, step_hist = {}, {}
for name in SETS:
    c = dict.fromkeys(LABELS, 0)
    h = []
    for t in TASKS:
        for s in range(SEEDS):
            r = run_agent(t["q"], FaultyPlanner(t, .3, s), SETS[name])
            c[outcome(r, t)] += 1
            h.append(r.steps)
    reasons[name] = c
    step_hist[name] = h
json.dump(dict(tasks=len(TASKS), seeds=SEEDS, runs_per_point=len(TASKS) * SEEDS, faults=FAULTS, success=table, mean_steps=steps,
               by_family=family, outcomes_at_30pct=reasons), open(OUT / "summary.json", "w"), indent=2)
print("runs per point", len(TASKS) * SEEDS, "reasons", reasons)
print("family", {k: [round(100 * x, 1) for x in v] for k, v in family.items()})

COLS = ["#9aa0a6", "#f4b400", "#12b5cb", "#d01884"]
fig, a = plt.subplots(figsize=(7.5, 4.3), constrained_layout=True)
for (n, v), c in zip(steps.items(), COLS):
    a.plot(100 * np.array(FAULTS), v, "o-", lw=2.2, c=c, label=n)
a.set(xlabel="Chance of a model mistake on each reply (%)", ylabel="Mean steps per task", title="Recovering from mistakes costs extra model calls")
a.grid(alpha=.25)
a.spines[["top", "right"]].set_visible(False)
a.legend(frameon=False)
fig.savefig(OUT / "steps_vs_mistakes.png", dpi=170)
plt.close(fig)

fig, a = plt.subplots(figsize=(8, 3.9), constrained_layout=True)
names = list(SETS)
left = np.zeros(len(names))
for lab, c in zip(LABELS, ["#188038", "#f4b400", "#d93025", "#9aa0a6"]):
    v = np.array([100 * reasons[n][lab] / (len(TASKS) * SEEDS) for n in names])
    a.barh(names, v, left=left, color=c, label=lab)
    left += v
a.invert_yaxis()
a.set(xlabel="Runs (%)", xlim=(0, 100), title="How runs end at a 30% mistake rate")
a.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.18))
a.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "how_runs_end.png", dpi=170)
plt.close(fig)

fig, a = plt.subplots(figsize=(7.5, 3.4), constrained_layout=True)
Z = 100 * np.array(list(family.values()))
im = a.imshow(Z, cmap="RdYlGn", vmin=60, vmax=100, aspect="auto")
a.set_xticks(range(len(FAULTS)), [f"{int(100 * f)}%" for f in FAULTS])
a.set_yticks(range(len(family)), list(family))
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        a.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center")
a.set(xlabel="Mistake rate", title="All guards: tasks solved (%) by task type")
fig.colorbar(im)
fig.savefig(OUT / "by_task_type.png", dpi=170)
plt.close(fig)

fig, a = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
bins = np.arange(.5, 13.5)
for n, c in (("All guards", "#d01884"), ("Show errors only", "#f4b400")):
    a.hist(step_hist[n], bins=bins, alpha=.65, color=c, label=n)
a.set(xlabel="Steps used in a run", ylabel="Runs", title="Steps per run at a 30% mistake rate")
a.spines[["top", "right"]].set_visible(False)
a.legend(frameon=False)
fig.savefig(OUT / "step_distribution.png", dpi=170)
plt.close(fig)

fig, a = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
for (n, v), c in zip(table.items(), ["#9aa0a6", "#f4b400", "#12b5cb", "#d01884"]):
    a.plot(100 * np.array(FAULTS), 100 * np.array(v), "o-", lw=2.4, c=c, label=n)
a.set(xlabel="Chance of a model mistake on each reply (%)", ylabel="Tasks solved correctly (%)", ylim=(0, 104),
      title=f"{len(TASKS)} engineering tasks, {SEEDS} seeds each")
a.grid(alpha=.25)
a.spines[["top", "right"]].set_visible(False)
a.legend(frameon=False)
fig.savefig(OUT / "robustness.png", dpi=170)
plt.close(fig)

t = wavelength(28)
r = run_agent(t["q"], FaultyPlanner(t, .35, 2))
lines = [f"$ python demo.py", "", f"Task: {t['q']}", ""]
for a_, b in r.trace:
    lines.append(f"  {a_}" + (f"  ->  {b}" if a_ != "final" else f"  {b}"))
lines += ["", f"Finished in {r.steps} steps, correct answer: {t['truth']:.4f}"]
import textwrap
w = [x for l in lines for x in (textwrap.wrap(l, 92, subsequent_indent="      ") or [""])]
fig = plt.figure(figsize=(10, .3 * len(w) + .3), facecolor="#0d1117")
for i, l in enumerate(w):
    col = "#7ee787" if l.startswith("$") or "final" in l[:9] else "#ff7b72" if "ERROR" in l or "bad reply" in l or "blocked" in l else "#c9d1d9"
    fig.text(.02, 1 - (i + 1.2) / (len(w) + 1), l, family="monospace", fontsize=11, color=col)
fig.savefig(OUT / "trace.png", dpi=170, facecolor=fig.get_facecolor())
print("\n".join(w))
