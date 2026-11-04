"""A tool-using AI agent in about 150 lines, with no framework.

The agent asks a language model what to do, runs the chosen tool, feeds the
result back, and repeats until the model gives a final answer. Language models
make mistakes, so the useful part is the set of guards around that loop.
"""
import ast, json, math, operator, re
from dataclasses import dataclass, field

CONSTANTS = {"speed_of_light_m_s": 299792458.0, "boltzmann_j_k": 1.380649e-23,
             "lora_bandwidth_hz": 125000.0, "esp32_clock_hz": 240e6, "arduino_uno_clock_hz": 16e6}
UNITS = {("ghz", "hz"): 1e9, ("mhz", "hz"): 1e6, ("khz", "hz"): 1e3, ("km", "m"): 1e3,
         ("m", "mm"): 1e3, ("kb", "b"): 1024.0, ("mb", "kb"): 1024.0, ("h", "s"): 3600.0, ("min", "s"): 60.0}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.USub: operator.neg}
FUNCS = {"log10": math.log10, "log2": math.log2, "sqrt": math.sqrt}


def calculator(expression):
    """Evaluates arithmetic safely by walking the syntax tree. It never calls eval()."""
    def run(n):
        if isinstance(n, ast.Expression):
            return run(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](run(n.left), run(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](run(n.operand))
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") in FUNCS and len(n.args) == 1:
            return FUNCS[n.func.id](run(n.args[0]))
        raise ValueError("unsupported expression")
    return float(run(ast.parse(expression, mode="eval")))


def lookup(name):
    if name not in CONSTANTS:
        raise KeyError(f"unknown constant, choose from {sorted(CONSTANTS)}")
    return CONSTANTS[name]


def convert(value, source, target):
    k = (source.lower(), target.lower())
    if k in UNITS:
        return float(value) * UNITS[k]
    if k[::-1] in UNITS:
        return float(value) / UNITS[k[::-1]]
    raise ValueError(f"cannot convert {source} to {target}")


TOOLS = {"calculator": calculator, "lookup": lookup, "convert": convert}


def parse_action(text, repair=True):
    """Reads the model's JSON reply. With repair on, it also accepts common damage."""
    try:
        d = json.loads(text)
    except ValueError:
        if not repair:
            raise
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        s = re.sub(r",\s*([}\]])", r"\1", m.group(0))
        try:
            d = json.loads(s)
        except ValueError:
            d = json.loads(s.replace("'", '"'))
    if not isinstance(d, dict):
        raise ValueError("reply is not an object")
    return d


@dataclass
class Guards:
    repair_json: bool = True
    show_errors: bool = True    # tell the model what went wrong so it can correct itself
    block_repeats: bool = True  # refuse the same failing call twice in a row


@dataclass
class Result:
    answer: object = None
    steps: int = 0
    trace: list = field(default_factory=list)
    stopped: str = ""


def run_agent(task, llm, guards=Guards(), max_steps=12):
    log = [{"role": "user", "content": task}]
    out = Result()
    last_failed = None
    for step in range(1, max_steps + 1):
        out.steps = step
        reply = llm.complete(log)
        try:
            act = parse_action(reply, guards.repair_json)
        except ValueError as e:
            out.trace.append(("bad reply", reply[:60]))
            if not guards.show_errors:
                out.stopped = "crashed on bad reply"
                return out
            log.append({"role": "tool", "content": "ERROR: reply was not valid JSON, resend it"})
            continue
        if "final" in act:
            out.answer = act["final"]
            out.trace.append(("final", str(act["final"])))
            return out
        tool, args = act.get("tool"), act.get("args", {})
        key = json.dumps([tool, args], sort_keys=True, default=str)
        if guards.block_repeats and key == last_failed:
            out.trace.append(("blocked repeat", tool))
            log.append({"role": "tool", "content": "ERROR: same failing call repeated, change something"})
            continue
        try:
            value = TOOLS[tool](**args)
            last_failed = None
            out.trace.append((f"{tool}({args})", value))
            log.append({"role": "tool", "content": value})
        except Exception as e:
            last_failed = key
            out.trace.append((f"{tool}({args})", f"ERROR {e}"))
            if guards.show_errors:
                log.append({"role": "tool", "content": f"ERROR: {e}"})
            else:
                out.stopped = "crashed on tool error"
                return out
    out.stopped = "step limit"
    return out
