"""Language model backends.

OllamaLLM talks to a local model. It has not been tested against a live server.
FaultyPlanner is a scripted stand-in that follows a correct plan but injects the
kinds of mistakes real models make, so the guards can be measured repeatably.
"""
import json, random, urllib.request


class OllamaLLM:
    SYSTEM = ('You are an agent. Reply with JSON only. To use a tool: {"tool": name, "args": {...}}. '
              'To finish: {"final": answer}. Tools: calculator(expression), lookup(name), '
              'convert(value, source, target).')

    def __init__(self, model="llama3.2", url="http://localhost:11434/api/chat"):
        self.model, self.url = model, url

    def complete(self, log):
        msgs = [{"role": "system", "content": self.SYSTEM}]
        msgs += [{"role": "user" if m["role"] == "tool" else m["role"], "content": str(m["content"])} for m in log]
        body = json.dumps({"model": self.model, "messages": msgs, "stream": False, "format": "json"}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["message"]["content"]


class FaultyPlanner:
    """Follows task["plan"]. On each reply, a mistake happens with probability fault."""

    def __init__(self, task, fault, seed):
        self.t, self.fault, self.r = task, fault, random.Random(seed)
        self.i, self.obs, self.stuck = 0, [], 0

    def complete(self, log):
        last = log[-1]
        if last["role"] == "tool":
            if str(last["content"]).startswith("ERROR"):
                pass
            elif self.pending:
                self.obs.append(last["content"])
                self.i += 1
        self.pending = False
        if self.i >= len(self.t["plan"]):
            return self._damage(json.dumps({"final": self.t["answer"](self.obs)}))
        tool, make = self.t["plan"][self.i]
        args = make(self.obs)
        if self.r.random() < self.fault:
            kind = self.r.choice(["hallucinated_tool", "bad_args"])
            self.pending = False
            if kind == "hallucinated_tool":
                return json.dumps({"tool": "web_search", "args": {"q": "help"}})
            return json.dumps({"tool": tool, "args": {"wrong_name": 1}})
        self.pending = True
        return self._damage(json.dumps({"tool": tool, "args": args}))

    def _damage(self, s):
        if self.r.random() >= self.fault:
            return s
        kind = self.r.choice(["prose", "trailing", "cut"])
        if kind == "prose":
            return "Sure! Here is my next step:\n```json\n" + s + "\n```"
        if kind == "trailing":
            return s[:-1] + ",}"
        self.pending = False
        return s[: len(s) // 2]

    pending = False
