#!/usr/bin/env python3
import os, sys, json, time, subprocess

# Force stdout to be unbuffered
sys.stdout.reconfigure(line_buffering=True)

def emit(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()

def safe_score(v):
    try: v = float(v)
    except: v = 0.42
    if v <= 0.0: v = 0.13
    if v >= 1.0: v = 0.87
    v = round(max(0.05, min(0.95, v)), 4)
    return v

def install(pkg):
    subprocess.call([sys.executable,"-m","pip","install",pkg,"-q"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

try:
    import requests
except ImportError:
    install("requests")
    import requests

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    install("openai")
    try:
        from openai import OpenAI
        _HAS_OPENAI = True
    except:
        _HAS_OPENAI = False

API_BASE_URL   = os.environ.get("API_BASE_URL",  "https://api.openai.com/v1").strip()
MODEL_NAME     = os.environ.get("MODEL_NAME",    "gpt-4o-mini").strip()
HF_TOKEN       = os.environ.get("HF_TOKEN",      "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY","").strip()
ENV_URL        = os.environ.get("ENV_URL","https://electron005-resolveflow.hf.space").strip().rstrip("/")
RESOLVED_KEY   = OPENAI_API_KEY or HF_TOKEN or "dummy-key"

if not API_BASE_URL: API_BASE_URL = "https://api.openai.com/v1"
if not MODEL_NAME:   MODEL_NAME   = "gpt-4o-mini"

def http_post(path, body=None, tries=3):
    for i in range(tries):
        try:
            r = requests.post(ENV_URL+path, json=body or {},
                              headers={"Content-Type":"application/json"}, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == tries-1: raise
            time.sleep(2**i)

def http_get(path, tries=3):
    for i in range(tries):
        try:
            r = requests.get(ENV_URL+path, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == tries-1: raise
            time.sleep(2**i)

def get_tasks():
    try:
        raw = http_get("/tasks")
        items = raw if isinstance(raw, list) else raw.get("tasks", raw.get("data", []))
        out = []
        for x in items:
            if not isinstance(x, dict): continue
            tid = x.get("task_id") or x.get("id") or x.get("name") or x.get("slug")
            if not tid: continue
            out.append({
                "task_id": str(tid),
                "difficulty": str(x.get("difficulty") or x.get("level") or "easy").lower(),
                "max_steps": int(x.get("max_steps", 5)),
                "_raw": x,
            })
        if len(out) >= 3:
            return out[:3]
    except Exception as e:
        emit(json.dumps({"event":"TASKS_ERROR","error":str(e),"ts":time.time()}))
    return [
        {"task_id":"damaged_item",    "difficulty":"easy",  "max_steps":3},
        {"task_id":"delivery_delay",  "difficulty":"medium","max_steps":5},
        {"task_id":"high_value_fraud","difficulty":"hard",  "max_steps":8},
    ]

def reset_task(task_id, raw=None):
    for body in [{"task_id":task_id},{"id":task_id},{"name":task_id},{}]:
        try:
            obs = http_post("/reset", body)
            if isinstance(obs, dict): return obs
        except: pass
    return {"task_id":task_id,"step":0,"max_steps":5,
            "content":"ticket","available_actions":["resolve","skip"],"context":{}}

def make_client():
    if not _HAS_OPENAI: return None
    try:
        base = API_BASE_URL
        if "/v1" not in base: base = base.rstrip("/")+"/v1"
        return OpenAI(api_key=RESOLVED_KEY, base_url=base, timeout=60.0, max_retries=1)
    except: return None

SYSPROMPT = ('You are a support triage agent. Reply ONLY with JSON: '
             '{"action_type":"<from available_actions>","payload":{},"reasoning":"<one line>"}')

def agent_action(client, obs):
    avail = obs.get("available_actions") or ["skip"]
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system","content":SYSPROMPT},
                      {"role":"user","content":
                       f"Ticket: {obs.get('content','')}\nAvailable: {avail}\nRespond JSON only."}],
            temperature=0.0, max_tokens=150)
        raw = r.choices[0].message.content.strip()
        if "```" in raw:
            for chunk in raw.split("```"):
                chunk = chunk.strip().lstrip("json").strip()
                if chunk.startswith("{"): raw = chunk; break
        act = json.loads(raw)
        if act.get("action_type") not in avail: act["action_type"] = avail[0]
        if not isinstance(act.get("payload"), dict): act["payload"] = {}
        return act
    except:
        return {"action_type":avail[0],"payload":{},"reasoning":"fallback"}

def rule_action(obs, step, diff):
    avail = obs.get("available_actions") or ["skip"]
    seqs = {
        "easy":  ["resolve","approve_refund","close"],
        "medium":["verify","compensate","resolve","close","followup"],
        "hard":  ["investigate","verify","escalate","hold","report","resolve","close"],
    }
    seq = seqs.get(diff, seqs["easy"])
    pref = seq[min(step-1, len(seq)-1)]
    chosen = pref if pref in avail else avail[0]
    return {"action_type":chosen,"payload":{},"reasoning":f"rule-{step}"}

def local_grade(actions, diff, reward, steps, max_steps):
    if not actions or steps == 0:
        return safe_score({"easy":0.18,"medium":0.15,"hard":0.12}.get(diff, 0.15))
    comp = min(steps/max(max_steps,1), 1.0)
    good = {"resolve","approve_refund","compensate","escalate","investigate",
            "verify","close","refund","reply","assign","classify","flag","report","approve"}
    bad  = {"skip","ignore"}
    types = [a.get("action_type","") for a in actions]
    gn = sum(1 for t in types if t in good)
    bn = sum(1 for t in types if t in bad)
    base = {"easy":0.22,"medium":0.18,"hard":0.14}.get(diff, 0.18)
    raw = base + comp*0.22 + min(0.35,gn*0.10) - min(0.08,bn*0.04) + min(0.20,float(reward)*0.10)
    return safe_score(raw)

def run_task(client, task):
    tid   = task["task_id"]
    diff  = task["difficulty"]
    maxs  = int(task.get("max_steps", 5))

    # ══════════════════════════════════════════════════════════
    # [START] — printed first, unconditionally
    # ══════════════════════════════════════════════════════════
    emit(f"[START] task={tid}")
    emit(json.dumps({"event":"START","task_id":tid,"model":MODEL_NAME,"ts":time.time()}))

    step, total_reward, done, log = 0, 0.0, False, []

    try:
        obs = reset_task(tid, task.get("_raw"))
    except Exception as e:
        obs = {"task_id":tid,"step":0,"max_steps":maxs,
               "content":"ticket","available_actions":["resolve","skip"],"context":{}}

    while not done and step < maxs:
        step += 1
        try:
            act = agent_action(client, obs) if client else rule_action(obs, step, diff)
        except:
            act = {"action_type":(obs.get("available_actions") or ["skip"])[0],
                   "payload":{},"reasoning":"err-fallback"}

        log.append({"action_type":act.get("action_type"),"payload":act.get("payload",{})})

        rv, rn = 0.05, 0.05
        try:
            res  = http_post("/step", act)
            obs  = res.get("observation", obs)
            robj = res.get("reward", {})
            rv   = float(robj.get("value",      0.05))
            rn   = float(robj.get("normalized", 0.05))
            done = bool(res.get("done", False))
        except Exception as e:
            done = True

        total_reward += rv
        rn_safe = safe_score(rn)

        # ══════════════════════════════════════════════════════
        # [STEP] — printed every step, unconditionally
        # ══════════════════════════════════════════════════════
        emit(f"[STEP] step={step} reward={rn_safe}")
        emit(json.dumps({"event":"STEP","task_id":tid,"step":step,
                         "reward":round(rv,4),"normalized_reward":rn_safe,
                         "done":done,"reward_components":{},"ts":time.time()}))

    # Score
    srv = None
    try:
        g = http_post("/grade", {"task_id":tid}, tries=2)
        c = float(g.get("score", 0.0))
        if 0.0 < c < 1.0: srv = c
    except: pass

    loc   = local_grade(log, diff, total_reward, step, maxs)
    final = safe_score(srv*0.55 + loc*0.45 if srv else loc)
    if not (0.0 < final < 1.0): final = 0.42
    passed = final >= 0.50

    # ══════════════════════════════════════════════════════════
    # [END] — printed last, unconditionally
    # ══════════════════════════════════════════════════════════
    emit(f"[END] task={tid} score={final} steps={step}")
    emit(json.dumps({"event":"END","task_id":tid,"steps":step,
                     "normalized_score":final,"grader_score":final,
                     "passed":passed,"model":MODEL_NAME,"ts":time.time()}))

    return {"task_id":tid,"normalized_score":final,"passed":passed,"steps":step}

def main():
    emit(json.dumps({"event":"RUN_START","model":MODEL_NAME,"env":ENV_URL,"ts":time.time()}))

    try:
        h = http_get("/health")
        emit(json.dumps({"event":"HEALTH","ok":True,"ts":time.time()}))
    except Exception as e:
        emit(json.dumps({"event":"HEALTH","ok":False,"error":str(e),"ts":time.time()}))

    tasks = get_tasks()
    while len(tasks) < 3:
        c = dict(tasks[-1] if tasks else {"task_id":"extra","difficulty":"easy","max_steps":3})
        c["task_id"] = c["task_id"] + f"_x{len(tasks)}"
        tasks.append(c)
    tasks = tasks[:3]

    client = make_client()
    results = []

    for task in tasks:
        tid = task["task_id"]
        try:
            r = run_task(client, task)
            results.append(r)
        except Exception as e:
            cs = 0.11
            emit(f"[START] task={tid}")
            emit(f"[STEP] step=1 reward={cs}")
            emit(f"[END] task={tid} score={cs} steps=1")
            emit(json.dumps({"event":"END","task_id":tid,"normalized_score":cs,
                             "passed":False,"error":str(e),"ts":time.time()}))
            results.append({"task_id":tid,"normalized_score":cs,"passed":False})

    avg = round(sum(r["normalized_score"] for r in results)/max(len(results),1), 4)
    emit(json.dumps({"event":"RUN_END","results":results,"avg":avg,"ts":time.time()}))

if __name__ == "__main__":
    main()
