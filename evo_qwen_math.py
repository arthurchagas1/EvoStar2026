# -*- coding: utf-8 -*-
"""
Evo-Reasoner++ (Coach-enabled, No-Gold) — QD Evolution of LLM Reasoning
Diagnosis patch + GEJ (Global End-of-Generation Judge), consensus-first & richer logs.

What's new vs your last version:
- GEJ is now truly consensus-first: panel sees ALL tokens + (optional) fused as a pseudo-candidate.
- Fuser cannot override consensus; fused token is just another candidate in the panel.
- Evidence guarantee for panel: synth from SOLVE/CHECK/PLAN or drop token if truly empty.
- Fallback order: panel -> majority per-cell -> majority population -> shortest consistent.
- Tie-break helper (units/percent mentions) nudges ties only; never overrides chosen token.
- Judge rubric tightened; WHY required.
- Per-task summary includes token supports (#cells, #cands), panel decision & WHY, chosen rep, fallbacks.
- Each message entry now has "answer_line": "FINAL_ANSWER: <token>" (blank if none) for easy tracking.

NEW (this patch):
- Context-safe generation: inputs are token-truncated to (context_window - max_new_tokens - margin) to avoid 4096 overflow.
- Robust decoding: we decode only the newly generated token slice (no fragile string prefix slicing).
- Panel cap: limit the number of distinct tokens shown to the panel via PANEL_MAX_TOKENS.

QWEN ADAPT:
- Models set to Qwen/Qwen2.5-7B-Instruct for AUTHOR and JUDGE.
- Default output directory set to 'evo_runs_qwen25'.
"""

import os, re, json, math, random, time, gc
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
from tqdm.auto import tqdm

os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:512,expandable_segments:true")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= Config =================
AUTHOR_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_MODEL_NAME  = "Qwen/Qwen2.5-7B-Instruct"
DTYPE             = torch.bfloat16

# Evolution settings
P, GENERATIONS, MATCHES_PER_GEN = 12, 4, 48
MUT_RATE, CROSS_RATE, ELITISM_KEEP = 0.7, 0.3, 4
SEED_AUTHOR, SEED_JUDGE = 42, 777

# Generation caps
MAX_NEW_TOKENS_AUTHOR_CEIL, MAX_NEW_TOKENS_JUDGE = 384, 48
TEMP_AUTHOR, TOPP_AUTHOR = 0.8, 0.95
TACTICS, DETAIL_LEVELS, CHECK_TARGETS = ["algebraic","constructive","divide-and-conquer","case-based"], [2,3,4], [1,2,3]
EARLY_STOP_DELTA = 0.02

# Paths / IO
OUT_DIR, GLOBAL_SUMMARY = "evo_runs_qwen25", "summary.json"
BENCH_PATH, RESUME_RUN = "data/math_benchmark.jsonl", True

# Batching / logs
BATCH_AUTHOR, BATCH_JUDGE, LOG_EVERY, REPORT_EVERY = 8, 6, 10, 10
JUDGE_ANSWER_MAX_CHARS, JUDGE_PROMPT_MAX_CHARS = 3000, 8000
FINAL_TAG = "FINAL_ANSWER:"

# Limit pending tasks this run
TASK_LIMIT = 100

# Ablations / knobs
USE_PERSISTENT_MAP_ELITES = True
USE_MAP_ELITES_FOR_SELECTION = True
USE_CROSSOVER = True
ENABLE_COST_NORMALIZATION = True
COST_NORM_LAMBDA = 0.02
COST_NORM_TAU = 200.0
FINAL_JUDGE_ON_FUSED = True  # used only as a tie sanity-check; fuser never overrides panel now

# Coach
COACH_ENABLED = True
COACH_USE_JUDGE_MODEL = True
COACH_TOPK_CHILDREN = 6
COACH_MAX_HINT_CHARS = 1600
COACH_INJECT_IN_INIT = True
COACH_INJECT_IN_MUTATE = True
COACH_INJECT_IN_CROSSOVER = True

# Messages logging
MSG_TRUNC = 4000

# === GEJ knobs (Global End-of-Generation Judge) ======================
USE_PANEL_FINAL_SELECTION = True          # master toggle
PANEL_TOPK_PER_TOKEN = 3                  # evidence reps per token
MAX_NEW_TOKENS_PANEL_JUDGE = 96           # panel judge output budget
PANEL_MAX_EVIDENCE_CHARS = 900            # per token evidence cap
PANEL_PARSE_RE = re.compile(r"(?im)^\s*FINAL_TOKEN\s*:\s*([^\s]+)\s*$")
# ---- NEW KNOB: cap number of distinct tokens shown to the panel ----
PANEL_MAX_TOKENS = 10
# ====================================================================

# ================== Canonicalization / Extraction ==================
_ws = re.compile(r"\s+")
num_pat = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
frac_pat = re.compile(r"(?:\\frac\{([^}]*)\}\{([^}]*)\})|(\d+)\s*/\s*(\d+)")
hash4_pat = re.compile(r"####\s*([^\n\r]+)")
th_en = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
th_eu = re.compile(r"[-+]?\d{1,3}(?:\.\d{3})+(?:,\d+)?")
dec_comma = re.compile(r"[-+]?\d+,\d+")
all_zero = re.compile(r"^[+-]?0+(?:[.,]0+)?$")
our_final_line_pat = re.compile(r"(?im)^\s*FINAL_ANSWER\s*:\s*(.+?)\s*$")
final_line_pat = re.compile(r"(?im)^\s*(?:final\s*answer|answer|result)\s*[:=]\s*(.+?)\s*$")
rhet_pat = re.compile(r"(?i)(?:thus|therefore|so|hence|consequently)[, ]+(?:the )?answer(?: is)?\s*[:=]?\s*([^\n\r]+)")
TAG_RE = re.compile(r"<(SETUP|PLAN|SOLVE|CHECK|ANSWER|META)>(.*?)</\1>", re.S|re.I)
mixed_frac = re.compile(r"^\s*([+-]?\d+)\s+(\d+)\s*/\s*(\d+)\s*$")
UNIT_WORDS = re.compile(r"\b(minutes?|hours?|secs?|seconds?|gb|mb|usd|\$|brl|percent|percentage|%|dollars?)\b", re.I)
OPS_IN_ANSWER = re.compile(r"[=+\-*/^]")

def _strip_latex(s:str)->str: 
    return s.replace("$","").replace("\\(","").replace("\\)","").replace("\u200b","")

def _has_units(s:str)->bool: return bool(UNIT_WORDS.search(s or ""))
def _has_ops(s:str)->bool: 
    s = (s or "").strip()
    if s.startswith("-"): s = s[1:]
    return bool(OPS_IN_ANSWER.search(s))

def _is_pure_numeric(s:str)->bool:
    if not s: return False
    if _has_units(s) or _has_ops(s): return False
    t = s.strip()
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?", t): return True
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?/[+-]?\d+(?:\.\d+)?", t): return True
    return False

def _norm_num(tok:str)->str:
    t = tok.strip()
    m = mixed_frac.fullmatch(t)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sign = -1 if a < 0 else 1; a = abs(a)
        return f"{sign*(a*c + b)}/{c}"
    m = re.fullmatch(r"\s*([^\s/]+)\s*/\s*([^\s/]+)\s*", t)
    if m: return f"{m.group(1).strip()}/{m.group(2).strip()}"
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", t) and ("," not in t):
        t = t.replace(".", "")
    elif "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".") if (t.rfind(",")>t.rfind(".")) else t.replace(",", "")
    elif "," in t:
        t = t.replace(",", ".") if dec_comma.fullmatch(t) else t.replace(",", "")
    t = t.replace("_","").strip()
    t = re.sub(r"\s*×\s*10\^([+-]?\d+)", r"e\1", t)
    return t

def _canon(token:Optional[str])->Optional[str]:
    if token is None: return None
    t = token.strip()
    if not t or not _is_pure_numeric(t): return None
    m = re.fullmatch(r"\s*([^\s/]+)\s*/\s*([^\s/]+)\s*", t)
    return (f"{_norm_num(m.group(1))}/{_norm_num(m.group(2))}" if m else _norm_num(t))

def _find_last_number(s:str)->Optional[str]:
    s0 = _strip_latex(s); matches=[]
    s0 = re.sub(r"\[CAND\s+\d+\]", " ", s0)
    for pat in (th_en, th_eu): matches += [(m.group(0).strip(), m.end()) for m in pat.finditer(s0) if not all_zero.fullmatch(m.group(0).strip())]
    matches += [(m.group(0).strip(), m.end()) for m in dec_comma.finditer(s0) if not all_zero.fullmatch(m.group(0).strip())]
    matches += [(m.group(0).strip(), m.end()) for m in num_pat.finditer(s0) if not all_zero.fullmatch(m.group(0).strip())]
    return max(matches, key=lambda x:x[1])[0] if matches else None

def _boxed_all(s:str)->List[str]: 
    return re.findall(r"\\boxed\{([^}]*)\}", s)

def _extract_token(s:str)->Optional[str]:
    if not s: return None
    m = our_final_line_pat.findall(s)
    if m and _is_pure_numeric(m[-1]): return _norm_num(m[-1])
    boxes = _boxed_all(s)
    if boxes:
        b = boxes[-1]
        mf = frac_pat.search(b)
        if mf:
            num, den = (mf.group(1), mf.group(2)) if mf.group(1) else (mf.group(3), mf.group(4))
            cand = f"{_norm_num(num)}/{_norm_num(den)}"
            return cand if _is_pure_numeric(cand) else None
        t = _find_last_number(b)
        t = _norm_num(t) if t else None
        return t if (t and _is_pure_numeric(t)) else None
    m = hash4_pat.search(s)
    if m and _is_pure_numeric(m.group(1).strip()): return _norm_num(m.group(1).strip())
    m = final_line_pat.findall(s)
    if m and _is_pure_numeric(m[-1]): return _norm_num(m[-1])
    m = list(rhet_pat.finditer(s))
    if m and _is_pure_numeric(m[-1].group(1)): return _norm_num(m[-1].group(1))
    mf = frac_pat.search(s)
    if mf:
        num, den = (mf.group(1), mf.group(2)) if mf.group(1) else (mf.group(3), mf.group(4))
        cand = f"{_norm_num(num)}/{_norm_num(den)}"
        if _is_pure_numeric(cand): return cand
    t = _find_last_number(s)
    t = _norm_num(t) if t else None
    return t if (t and _is_pure_numeric(t)) else None

def _to_float(token:Optional[str])->Optional[float]:
    if not token: return None
    t = _canon(token)
    if t is None: return None
    if re.fullmatch(r"[^\s/]+/[^\s/]+", t):
        a,b = t.split("/",1)
        try: bb = float(b); return float(a)/bb if bb!=0 else None
        except: return None
    try: return float(t)
    except: return None

def _last_line(text:str)->str:
    for ln in reversed(text.splitlines()):
        if ln.strip(): return ln.strip()
    return ""

def _answer_line_from_text(text:str)->str:
    tok = _canon(get_sections(text).get("ANSWER","").strip()) or _canon(_extract_token(text or ""))
    return f"{FINAL_TAG} {tok}" if tok else f"{FINAL_TAG} "

# ========= Task pre-normalization =========
def normalize_task(q:str)->str:
    q = re.sub(r"\bHow\s+load\b", "How long", q, flags=re.I)
    q = q.replace("mins", "minutes")
    return q

# =============== Strict Schema Validation & Repair ===============
ALLOWED_TAGS = {"SETUP","PLAN","SOLVE","CHECK","ANSWER","META"}

def _has_unknown_or_nested(text:str)->bool:
    for m in re.finditer(r"<([A-Z]+)[^>]*>", text):
        if m.group(1).upper() not in ALLOWED_TAGS: return True
    stripped = TAG_RE.sub("", text)
    return "<" in stripped or ">" in stripped

def is_valid_tagged_answer_strict(text:str)->bool:
    secs = {m.group(1).upper(): m.group(2) for m in TAG_RE.finditer(text)}
    if set(secs.keys()) != ALLOWED_TAGS: return False
    if _has_unknown_or_nested(text): return False
    ans = secs.get("ANSWER","").strip()
    return bool(_canon(ans))

REPAIR_PROMPT_STRICT = """You must rewrite EXACTLY using ONLY these tags once each, in this order:
<SETUP>...</SETUP>
<PLAN>...</PLAN>
<SOLVE>...</SOLVE>
<CHECK>...</CHECK>
<ANSWER>...</ANSWER>
<META>...</META>
Rules:
- No other tags, no nesting.
- <ANSWER> must be ONLY a bare number or fraction (no words/units).
- Keep content minimal and faithful.

Task:
{task}

Previous output:
{prev}
"""

# ================= Utils (loading/generation/logging) =================
def set_global_seed(seed:int):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _dev(m)->torch.device:
    try: return m.device
    except:
        try: return next(m.parameters()).device
        except: return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache()

def move_model(model, device:str):
    if _dev(model).type != device: model.to(device)

def swap_active_models(author_model, judge_model, author_on_gpu:bool):
    if author_on_gpu:
        move_model(judge_model,"cpu"); clear_cuda_cache(); move_model(author_model,"cuda")
    else:
        move_model(author_model,"cpu"); clear_cuda_cache(); move_model(judge_model,"cuda")

def try_load_with_attn(model_name:str, attn_impl:str):
    return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=DTYPE, attn_implementation=attn_impl)

def load_llm_bf16(name:str, seed:int, initial_device:str):
    set_global_seed(seed)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model, used = None, None
    for attn in ("flash_attention_2","sdpa","eager"):
        try:
            model = try_load_with_attn(name, attn); tqdm.write(f"[load_llm] attn={attn}"); used=attn; break
        except Exception as e:
            tqdm.write(f"[load_llm] attn={attn} failed: {e}")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, dtype=DTYPE); used="default"
    move_model(model, "cuda" if (initial_device=="cuda" and torch.cuda.is_available()) else "cpu")
    torch.set_float32_matmul_precision("high"); model.config.use_cache=True; model.generation_config.pad_token_id=tok.eos_token_id
    try: model.generation_config.top_k=None
    except: pass
    return tok, model, used

def apply_chat_template(tokenizer, user_text:str):
    try:
        if hasattr(tokenizer,"apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template([{"role":"user","content":user_text}], add_generation_prompt=True, tokenize=False)
    except: pass
    return user_text

def middle_truncate_text(s:str, cap:int)->str:
    if len(s)<=cap: return s
    head=cap//2; tail=cap-head-15
    return s[:head]+"\n...[TRUNCATED]...\n"+s[-tail:]

# ---- Context-safe helper ----
def _ctx_len(model, tok=None) -> int:
    # Prefer model config if available, else tokenizer model_max_length (guard huge sentinel), else 4096
    cfg_max = int(getattr(model.config, "max_position_embeddings", 0) or getattr(model.config, "max_sequence_length", 0) or 0)
    if tok is not None:
        tmax = int(getattr(tok, "model_max_length", 0) or 0)
        if tmax > 0 and tmax < 10**6:
            cfg_max = max(cfg_max, tmax)
    return cfg_max if cfg_max > 0 else 4096

def robust_generate_batch(tok, model, prompts:List[str], max_new:int, temp:Optional[float], top_p:Optional[float], seed:int, bs:int, allow_cpu_fallback:bool=True)->Tuple[List[str],Dict[str,Any]]:
    """
    Safe batched generation:
      - Token-level truncation to keep (input_len + max_new + margin) <= context_window
      - Robust extraction of only the generated tokens by slicing on input lengths per row
    """
    set_global_seed(seed); outs=[]; secs_total=0.0
    CTX_MARGIN = 8

    def run_one(chunk, device, use_cache, cap):
        texts=[apply_chat_template(tok,p) for p in chunk]
        ctx = _ctx_len(model, tok)
        budget = max(64, ctx - int(cap) - CTX_MARGIN)

        # Tokenize with truncation to budget
        inp = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=budget).to(device)
        with torch.no_grad():
            t0=time.time()
            out = model.generate(
                **inp,
                max_new_tokens=cap,
                do_sample=(temp is not None),
                temperature=(temp if temp is not None else 1.0),
                top_p=(top_p if temp is not None else 1.0),
                top_k=None,
                pad_token_id=tok.eos_token_id,
                use_cache=use_cache
            )
            if device.type=="cuda": torch.cuda.synchronize()
            dt=time.time()-t0

        # Slice out ONLY new tokens per example (avoid brittle string prefixing)
        gen_texts=[]
        input_ids = inp["input_ids"]
        for i in range(out.size(0)):
            # true length without padding
            in_len = int((input_ids[i] != tok.pad_token_id).sum().item())
            gen_ids = out[i, in_len:]
            gen_texts.append(tok.decode(gen_ids, skip_special_tokens=True).strip())
        return gen_texts, dt

    def chunks(a,n):
        for i in range(0,len(a),n): yield a[i:i+n]

    with torch.inference_mode():
        i=0
        while i<len(prompts):
            micro=prompts[i:i+bs]; success=False
            tries = [
                ("gpu",bs,True,max_new),("gpu",max(1,bs//2),True,max_new),
                ("gpu",max(1,bs//2),False,max_new),("gpu",1,False,max_new),
                ("gpu",1,False,max(12,max_new//2)),("gpu",1,False,12),
                ("cpu",min(2,len(micro)),False,12) if allow_cpu_fallback else None
            ]
            for t in tries:
                if t is None: continue
                where, this_bs, use_cache, cap = t
                try:
                    if where=="gpu":
                        if _dev(model).type!="cuda": move_model(model,"cuda"); torch.cuda.synchronize()
                        acc, sec = [], 0.0
                        for sub in chunks(micro, this_bs):
                            g, s = run_one(sub, torch.device("cuda"), use_cache, cap); acc+=g; sec+=s
                    else:
                        move_model(model,"cpu"); acc, sec = [], 0.0
                        for sub in chunks(micro, this_bs):
                            g, s = run_one(sub, torch.device("cpu"), False, 12); acc+=g; sec+=s
                        move_model(model,"cuda"); torch.cuda.synchronize()
                    outs += acc; secs_total += sec; success=True; break
                except torch.cuda.OutOfMemoryError:
                    clear_cuda_cache(); continue
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e): clear_cuda_cache(); continue
                    else: raise
            if not success: raise RuntimeError("OOM backoff exhausted.")
            i += bs
    return outs, {"seconds": secs_total}

# ---------- Logging helper: always attach answer_line ----------
def _append_msg(messages:List[Dict[str,Any]], entry:Dict[str,Any])->None:
    out_text = entry.get("output","") or entry.get("response","") or ""
    entry["answer_line"] = _answer_line_from_text(out_text) if out_text else f"{FINAL_TAG} "
    messages.append(entry)

# ================= Structures & Prompts =================
@dataclass
class Individual:
    id:str; text:str; tactic:str; detail:int; target_checks:int
    tokens:int=0; checks_found:int=0; length_bin:str=""; checks_bin:str=""; elo:float=1000.0
    meta:Dict=field(default_factory=dict); gen_stats:Dict[str,Any]=field(default_factory=dict)

def get_sections(text:str)->Dict[str,str]:
    return {m.group(1).upper(): m.group(2).strip() for m in TAG_RE.finditer(text)}

def count_checks_from_check_block(s:str)->int:
    if not s: return 0
    lines=[ln.strip() for ln in s.splitlines() if ln.strip()]
    bullets=sum(1 for ln in lines if re.match(r"^(-|\*|\d+[\.\)])\s", ln))
    return bullets if bullets>0 else max(1, len(lines)//2)

def length_bin_from_tokens(n:int)->str: return "short" if n<250 else ("medium" if n<550 else "long")
def checks_bin_from_count(n:int)->str: return "low" if n<=1 else ("medium" if n==2 else "high")

_AUTHOR_HEADER_BASE = """You are the AUTHOR LLM.
Output ONLY these tags (English) once each, in this order:
<SETUP>...</SETUP>
<PLAN>...</PLAN>
<SOLVE>...</SOLVE>
<CHECK>...</CHECK>
<ANSWER>...</ANSWER>
<META>tactic, detail level, #checks (short)</META>
Rules:
- No other tags, no nesting.
- In <ANSWER>, output ONLY the final bare numeric value (no words/units).
- Keep it concise; compute the final number cleanly.

Task:
{task}

"""

def build_author_header_with_guidance(task:str, guidance:Optional[str])->str:
    task = normalize_task(task)
    base = _AUTHOR_HEADER_BASE.format(task=task)
    if guidance:
        g = guidance.strip()
        if len(g)>COACH_MAX_HINT_CHARS:
            g = g[:COACH_MAX_HINT_CHARS//2] + "\n...[TRUNCATED COACH]...\n" + g[-COACH_MAX_HINT_CHARS//2:]
        return base + "<!-- COACH HINTS (no numbers):\n" + g + "\n-->\n\n"
    return base

def make_author_prompts(task:str, combos:List[Tuple[str,int,int]], guidance:Optional[str])->List[str]:
    header = build_author_header_with_guidance(task, guidance if COACH_INJECT_IN_INIT and COACH_ENABLED else None)
    return [header + f"- Target tactic: {ta}\n- Detail level: {de}\n- Target #checks: {ck}\n\n{task}"
            for (ta,de,ck) in combos]

JUDGE_PROMPT = """You are the JUDGE LLM. Compare TWO tagged answers to the same task.
Use logic/math consistency; ignore verbosity except ties.
Penalize if <CHECK> does not verify the <ANSWER> math.

Output exactly:
WINNER: A | WINNER: B | WINNER: TIE
WHY: <≤ 10 words>  (REQUIRED)

Task:
{task}

==== Answer A ====
{A}

==== Answer B ====
{B}

Decide strictly; use TIE only if truly equivalent.
"""
WIN_RE = re.compile(r"WINNER:\s*(A|B|TIE)", re.I)

# === GEJ panel judge prompt =========================================
PANEL_JUDGE_PROMPT = """You are the JUDGE LLM. See the problem and multiple candidate NUMERIC answers with short evidence.
Choose the SINGLE best numeric token that matches the problem. Ignore verbosity; favor correct math and consistency.

Problem:
{task}

Candidates:
{cands}

Output EXACTLY:
FINAL_TOKEN: <numeric or fraction>
WHY: <≤ 12 words>
"""
# =====================================================================

def clip_answer_for_judge(ans:str, per:int=JUDGE_ANSWER_MAX_CHARS)->str:
    secs=get_sections(ans); out=[]
    for name in ["SETUP","PLAN","SOLVE","CHECK","ANSWER","META"]:
        if name in secs:
            body=secs[name]; body=middle_truncate_text(body, per//2 if name in ("SOLVE","CHECK") else per//4)
            out.append(f"<{name}>{body}</{name}>")
    return "\n".join(out) if out else middle_truncate_text(ans, per)

# ================= Validators & Repair =================
def is_valid_tagged_answer(text:str)->bool:
    return is_valid_tagged_answer_strict(text)

def repair_or_synthesize(at, am, task:str, bad_text:str, seed:int)->Tuple[str,str]:
    prompts = [REPAIR_PROMPT_STRICT.format(task=task, prev=bad_text)]
    gens, _ = robust_generate_batch(at, am, prompts, 196, None, None, seed, 1, False)
    g = gens[0]
    if is_valid_tagged_answer_strict(g): return g, "strict_repair_ok"
    salvage = _extract_token(bad_text)
    tok = _canon(salvage) or ""
    g2 = (
        "<SETUP></SETUP>\n<PLAN></PLAN>\n<SOLVE></SOLVE>\n<CHECK></CHECK>\n"
        f"<ANSWER>{tok}</ANSWER>\n<META>auto-repaired</META>"
    )
    return g2, ("salvage_numeric" if tok else "salvage_empty")

# ================= Elo, Tournament, MAP-Elites =================
def elo_update(ra, rb, score_a, k=16.0):
    ea = 1.0/(1.0+10**((rb-ra)/400.0)); eb=1-ea
    return ra + k*(score_a-ea), rb + k*((1-score_a)-eb)

def _cost_penalty(len_a:int, len_b:int)->float:
    if not ENABLE_COST_NORMALIZATION: return 0.0
    diff = float(len_a - len_b)
    return COST_NORM_LAMBDA * math.tanh(diff / max(1.0, COST_NORM_TAU))

def swiss_near_pairs(idx:List[int], elos:List[float], n:int, rng:random.Random)->List[Tuple[int,int]]:
    order=sorted(idx, key=lambda i:(elos[i], rng.random())); pairs=[]; i=0
    while len(pairs)<n:
        a=order[i%len(order)]; b=order[(i+1)%len(order)]; pairs.append((a,b)); i+=2
    return pairs

def run_pairwise_tournament(task:str, inds:List[Individual], jt, jm, matches:int, seed:int, log_every:int, messages:List[Dict[str,Any]]):
    if len(inds)<2 or matches<=0: 
        return {"upsets":0,"matches":0,"ties":0,"judge_cost_tokens":0,"judge_seconds":0.0}
    rng=random.Random(seed); idx=list(range(len(inds))); elos=[x.elo for x in inds]; pairs=swiss_near_pairs(idx, elos, matches, rng)
    prompts, order=[],[]
    for (i,j) in pairs:
        ia,ib=inds[i],inds[j]; A0=re.sub(r"<META>.*?</META>","",ia.text,flags=re.S|re.I).strip()
        B0=re.sub(r"<META>.*?</META>","",ib.text,flags=re.S|re.I).strip()
        if rng.random()<0.5: A_raw,B_raw,swap=A0,B0,False
        else: A_raw,B_raw,swap=B0,A0,True
        A_=clip_answer_for_judge(A_raw); B_=clip_answer_for_judge(B_raw); pr=JUDGE_PROMPT.format(task=task,A=A_,B=B_)
        if len(pr)>JUDGE_PROMPT_MAX_CHARS: pr=middle_truncate_text(pr, JUDGE_PROMPT_MAX_CHARS)
        prompts.append(pr); order.append((i,j,swap,ia.elo,ib.elo))
    outs,stats = robust_generate_batch(jt, jm, prompts, MAX_NEW_TOKENS_JUDGE, None, None, seed, BATCH_JUDGE, True)

    upset=0; ties=0
    for k,(i,j,swap,pre_a,pre_b) in enumerate(order,1):
        ia,ib=inds[i],inds[j]
        m=WIN_RE.search(outs[k-1]); w="TIE" if not m else m.group(1).upper()
        if w=="A" and swap: w="B"
        elif w=="B" and swap: w="A"
        if w=="TIE": ties += 1
        base_score_a = 1.0 if w=="A" else (0.5 if w=="TIE" else 0.0)
        len_a, len_b = ia.tokens, ib.tokens
        score_a = max(0.0, min(1.0, base_score_a - _cost_penalty(len_a, len_b)))
        ia.elo, ib.elo = elo_update(ia.elo, ib.elo, score_a)
        if (w=="A" and pre_a<pre_b) or (w=="B" and pre_b<pre_a): upset+=1
        if log_every and k%log_every==0: tqdm.write(f"[JUDGE] {k}/{len(pairs)} -> {w}")
        _append_msg(messages, {"role":"judge","prompt":middle_truncate_text(prompts[k-1],MSG_TRUNC),"output":middle_truncate_text(outs[k-1],MSG_TRUNC)})
    return {"upsets":upset,"matches":len(pairs),"ties":ties,"judge_cost_tokens":0,"judge_seconds":stats["seconds"]}

class MapElites:
    def __init__(self):
        self.archive:Dict[Tuple[str,str,str],Individual]={}
    def key(self,ind:Individual): return (ind.tactic, ind.length_bin, ind.checks_bin)
    def consider(self,ind:Individual):
        k=self.key(ind); b=self.archive.get(k)
        if b is None or ind.elo>b.elo: self.archive[k]=ind
    def consider_all(self, inds:List[Individual]):
        for x in inds: self.consider(x)
    def champions(self)->List[Individual]:
        return sorted(self.archive.values(), key=lambda x:x.elo, reverse=True)
    def per_cell_elites(self, k_max:int)->List[Individual]:
        return self.champions()[:k_max]

# ================= Operators (init/mutate/crossover/synthesize/coach) =================
def _ind_from_text(tok, text:str, tactic:str, det:int, chk:int, ind_id:str, dt:float, elo_seed:float=1000.0, parent_text:Optional[str]=None)->Individual:
    secs=get_sections(text); n_checks=count_checks_from_check_block(secs.get("CHECK","")); n_tok=len(tok.encode(text, add_special_tokens=False))
    ind = Individual(ind_id, text, tactic, det, chk, tokens=n_tok, checks_found=n_checks,
                     length_bin=length_bin_from_tokens(n_tok), checks_bin=checks_bin_from_count(n_checks),
                     elo=float(elo_seed), gen_stats={"author_seconds":dt})
    if parent_text:
        pa = get_sections(parent_text).get("ANSWER","").strip()
        ca = get_sections(text).get("ANSWER","").strip()
        if _canon(pa) and _canon(pa)==_canon(ca) and abs(len(text)-len(parent_text))<=max(20,int(0.03*len(parent_text))):
            ind.tokens += 50
            ind.meta["repeat_penalty"]=True
    return ind

def init_population(at, am, task:str, P:int, seed_base:int, guidance:Optional[str], messages:List[Dict[str,Any]])->List[Individual]:
    combos=[(t,d,c) for t in TACTICS for d in DETAIL_LEVELS for c in CHECK_TARGETS]; random.shuffle(combos); combos=combos[:P]
    prompts=make_author_prompts(task, combos, guidance); cap=max(128+48*d+32*c for _,d,c in combos)
    gens,st=robust_generate_batch(at, am, prompts, cap, TEMP_AUTHOR, TOPP_AUTHOR, seed_base, BATCH_AUTHOR, False)
    dt=st["seconds"]/max(1,len(gens)); out=[]
    for i,(g,(tac,det,chk),pr) in enumerate(zip(gens,combos,prompts)):
        _append_msg(messages, {"role":"author","type":"init","prompt":middle_truncate_text(pr,MSG_TRUNC),"output":middle_truncate_text(g,MSG_TRUNC)})
        if not is_valid_tagged_answer(g):
            g2, reason = repair_or_synthesize(at, am, task, g, seed_base+100+i)
            _append_msg(messages, {"role":"author","type":"repair","reason":reason,"output":middle_truncate_text(g2,MSG_TRUNC)})
            g = g2
        x=_ind_from_text(at,g,tac,det,chk,f"gen0_{i:02d}",dt,1000.0)
        x.meta["gen"]=0; out.append(x)
    return out

def _mut_prompt(task,parent:Individual,tactic:str,detail:int,checks:int,mtype:str,guidance:Optional[str])->str:
    rules=["Apply minimal local edits. Keep the exact tag structure.",
           f"Target tactic: {tactic}.", f"Detail level: {detail}.", f"Target #checks: {checks}.",
           "In <ANSWER>, output ONLY the final bare numeric value (no words/units).",
           "Do not output any headings/markers outside the 6 tags."]
    if mtype=="rewrite_solve": rules.append("Rewrite ONLY <SOLVE> to be more direct.")
    if mtype=="insert_check": rules.append("Add 1 targeted verification in <CHECK> for a key step.")
    if mtype=="reorder_steps": rules.append("Reorder <PLAN> steps for logical flow; keep content minimal.")
    if mtype=="standardize_notation": rules.append("Standardize symbols/units across sections without changing meaning.")
    instr="- "+"\n- ".join(rules)
    coach_block = f"\nCoach guidance for THIS generation:\n{guidance}\n" if (COACH_ENABLED and COACH_INJECT_IN_MUTATE and guidance) else ""
    return ("You are the AUTHOR LLM.\n"
            "Produce a NEW COMPLETE answer (all tags), changing only as required.\n"
            "No text outside the tags.\n\n"
            f"Task:\n{normalize_task(task)}\n\n"
            f"Mutation instructions:\n{instr}\n"
            f"{coach_block}\n"
            f"Current answer:\n{parent.text}\n")

def author_mutate_batch(at, am, task:str, parents:List[Individual], ids:List[str], seed:int, guidance:Optional[str], messages:List[Dict[str,Any]])->List[Individual]:
    rng=random.Random(seed); prompts,meta,caps=[],[],[]
    for cid,p in zip(ids,parents):
        tactic=p.tactic; mtype=rng.choice(["tactic","detail+","detail-","checks+","checks-","rewrite_solve","insert_check","reorder_steps","standardize_notation"])
        detail=max(1,min(5,p.detail+(1 if mtype=="detail+" else -1 if mtype=="detail-" else 0)))
        checks=max(1,p.target_checks+(1 if mtype=="checks+" else -1 if mtype=="checks-" else 0))
        if mtype=="tactic":
            others=[t for t in TACTICS if t!=tactic]
            if others: tactic=rng.choice(others)
        pr=_mut_prompt(task,p,tactic,detail,checks,mtype,guidance)
        prompts.append(pr); meta.append((cid,tactic,detail,checks,p.elo,p.text)); caps.append(128+48*detail+32*checks)
    gens,st=robust_generate_batch(at, am, prompts, max(caps) if caps else 192, TEMP_AUTHOR, TOPP_AUTHOR, seed, BATCH_AUTHOR, False)
    dt=st["seconds"]/max(1,len(gens)); out=[]
    for i,(g,(cid,tac,det,chk,elo_seed,parent_text),pr) in enumerate(zip(gens,meta,prompts)):
        _append_msg(messages, {"role":"author","type":"mutate","prompt":middle_truncate_text(pr,MSG_TRUNC),"output":middle_truncate_text(g,MSG_TRUNC)})
        if not is_valid_tagged_answer(g):
            g2, reason = repair_or_synthesize(at, am, task, g, seed+100+i)
            _append_msg(messages, {"role":"author","type":"repair","reason":reason,"output":middle_truncate_text(g2,MSG_TRUNC)})
            g = g2
        out.append(_ind_from_text(at,g,tac,det,chk,cid,dt,float(elo_seed),parent_text))
    return out

def author_crossover_batch(at, am, task:str, As:List[Individual], Bs:List[Individual], ids:List[str], seed:int, guidance:Optional[str], messages:List[Dict[str,Any]])->List[Individual]:
    prompts,meta,caps=[],[],[]
    coach_block = f"\nCoach guidance for THIS generation:\n{guidance}\n" if (COACH_ENABLED and COACH_INJECT_IN_CROSSOVER and guidance) else ""
    for a,b,cid in zip(As,Bs,ids):
        pr=("You are the AUTHOR LLM.\n"
            "Generate a CHILD by homologous crossover of two parents. Output FULL answer with tags.\n"
            "- Merge <PLAN>; interleave/dedup <CHECK>; ensure <SOLVE> consistency.\n"
            "- <ANSWER> ONLY bare number (no words/units). No text outside tags.\n\n"
            f"Task:\n{normalize_task(task)}\n\n"
            f"{coach_block}\n"
            f"--- PARENT A ---\n{a.text}\n\n--- PARENT B ---\n{b.text}\n")
        prompts.append(pr)
        tactic=random.choice([a.tactic,b.tactic]); detail=int(round((a.detail+b.detail)/2)); checks=max(1,int(round((a.target_checks+b.target_checks)/2)))
        meta.append((cid,tactic,detail,checks,0.5*(a.elo+b.elo),a.text))
        caps.append(128+48*detail+32*checks)
    gens,st=robust_generate_batch(at, am, prompts, max(caps) if caps else 224, TEMP_AUTHOR, TOPP_AUTHOR, seed, BATCH_AUTHOR, False)
    dt=st["seconds"]/max(1,len(gens)); out=[]
    for i,(g,(cid,tac,det,chk,elo_seed,parent_text),pr) in enumerate(zip(gens,meta,prompts)):
        _append_msg(messages, {"role":"author","type":"crossover","prompt":middle_truncate_text(pr,MSG_TRUNC),"output":middle_truncate_text(g,MSG_TRUNC)})
        if not is_valid_tagged_answer(g):
            g2, reason = repair_or_synthesize(at, am, task, g, seed+200+i)
            _append_msg(messages, {"role":"author","type":"repair","reason":reason,"output":middle_truncate_text(g2,MSG_TRUNC)})
            g = g2
        out.append(_ind_from_text(at,g,tac,det,chk,cid,dt,float(elo_seed),parent_text))
    return out

# ===== Coach: sanitize to remove numeric literals / final answers =====
def _sanitize_guidance(g:str)->Optional[str]:
    if not g: return None
    out=[]
    for ln in g.splitlines():
        s = ln.strip()
        if re.search(r"\b(answer|result)\b", s, re.I): continue
        if "=" in s or "≈" in s or ":" in s:
            if re.match(r"^\s*\d+[.)]\s+\D+$", s): out.append(s); continue
            continue
        if re.search(r"\d", s):
            if re.match(r"^\s*\d+[.)]\s+\D+$", s): out.append(s); continue
            continue
        out.append(s)
    txt="\n".join([x for x in out if x])
    if not txt or (len(re.findall(r"\d", txt))>0): return None
    return txt[:COACH_MAX_HINT_CHARS]

COACH_PROMPT = """You are a COACH LLM. Write ONE short guidance block (no numbers, no final answers).
Focus ONLY on symbolic sub-steps to compute and checks to perform (units, signs, case splits).
Return:
<GUIDANCE> ... </GUIDANCE>

Problem:
{task}

Current generation candidates (tagged, excerpted):
{cands}
"""

def _mk_guidance_excerpt(ind: Individual, per:int=800)->str:
    s = get_sections(ind.text)
    def trunc(t, cap):
        if len(t) <= cap: return t
        head = cap // 2; tail = cap - head - 15
        return t[:head] + "\n...[TRUNCATED]...\n" + t[-tail:]
    return (
        f"<PLAN>{trunc(s.get('PLAN',''), per//3)}</PLAN>\n"
        f"<SOLVE>{trunc(s.get('SOLVE',''), per//2)}</SOLVE>\n"
        f"<CHECK>{trunc(s.get('CHECK',''), per//3)}</CHECK>\n"
        f"<ANSWER>{trunc(s.get('ANSWER',''), 64)}</ANSWER>"
    )

def coach_generate_hint(task_text:str, generation_children:List[Individual], at, am, jt, jm, seed:int, messages:List[Dict[str,Any]])->Tuple[Optional[str], float]:
    if not COACH_ENABLED or not generation_children:
        return None, 0.0
    kids = sorted(generation_children, key=lambda x: x.elo, reverse=True)[:COACH_TOPK_CHILDREN]
    excerpts = "\n\n".join(_mk_guidance_excerpt(ind) for ind in kids)
    pr = COACH_PROMPT.format(task=normalize_task(task_text), cands=excerpts)
    tok, mdl = (jt, jm) if COACH_USE_JUDGE_MODEL else (at, am)
    outs, st = robust_generate_batch(tok, mdl, [pr], 196, None, None, seed, 1, True)
    raw = outs[0] if outs else ""
    _append_msg(messages, {"role":"coach","prompt":middle_truncate_text(pr,MSG_TRUNC),"output":middle_truncate_text(raw,MSG_TRUNC)})
    m = re.search(r"(?is)<GUIDANCE>(.*?)</GUIDANCE>", raw)
    hint = _sanitize_guidance(m.group(1).strip() if m else raw)
    return ("<GUIDANCE>\n"+hint+"\n</GUIDANCE>" if hint else None), float(st.get("seconds",0.0))

# ================= GEJ helpers (token clustering, evidence, panel) =================
def _answer_token_of(ind: Individual)->Optional[str]:
    sec = get_sections(ind.text).get("ANSWER","").strip()
    tok = _canon(sec) or _canon(_extract_token(ind.text))
    return tok

def _cluster_by_token(inds: List[Individual])->Dict[str, List[Individual]]:
    buckets: Dict[str, List[Individual]] = {}
    for ind in inds:
        tok = _answer_token_of(ind)
        if not tok: continue
        buckets.setdefault(tok, []).append(ind)
    return buckets

def _evidence_excerpt(ind: Individual, cap:int=PANEL_MAX_EVIDENCE_CHARS)->str:
    s = get_sections(ind.text)
    parts = []
    for k in ["PLAN","SOLVE","CHECK"]:
        if k in s and s[k]:
            parts.append(f"<{k}>{s[k]}</{k}>")
    ev = "\n".join(parts).strip()
    if not ev:
        # synth fallback: include anything minimal from text
        ev = middle_truncate_text(ind.text, cap)
    return middle_truncate_text(ev, cap)

def _build_panel_prompt(task:str, buckets: Dict[str, List[Individual]])->Tuple[str, List[str], Dict[str,Tuple[int,int]]]:
    # order tokens by (#cands desc, best Elo desc)
    ordering = []
    support_map: Dict[str,Tuple[int,int]] = {}  # token -> (#cells, #cands)
    for tok, lst in buckets.items():
        best_elo = max((x.elo for x in lst), default=0.0)
        # count distinct cells supporting this token
        cells = set((x.tactic, x.length_bin, x.checks_bin) for x in lst)
        support_map[tok] = (len(cells), len(lst))
        ordering.append((tok, len(lst), best_elo))
    ordering.sort(key=lambda t:(-t[1], -t[2], t[0]))
    ordering = ordering[:PANEL_MAX_TOKENS]  # <-- cap here

    # build candidates block, skipping tokens with truly empty evidence (after synth)
    cands_blocks = []
    tokens_ordered: List[str] = []
    for tok, _, _ in ordering:
        reps = sorted(buckets[tok], key=lambda x:x.elo, reverse=True)[:PANEL_TOPK_PER_TOKEN]
        evs_list = []
        for r in reps:
            evs_list.append(_evidence_excerpt(r))
        evs = "\n---\n".join([e for e in evs_list if e.strip()])
        if not evs.strip():
            continue  # starve tokens with no usable evidence
        block = f"[TOKEN {tok}]  (votes={support_map[tok][1]}, cells={support_map[tok][0]})\nEVIDENCE:\n{evs}\n"
        cands_blocks.append(block)
        tokens_ordered.append(tok)

    prompt = PANEL_JUDGE_PROMPT.format(task=normalize_task(task), cands="\n\n".join(cands_blocks) if cands_blocks else "<no-usable-evidence>")
    return prompt, tokens_ordered, support_map

def _unit_tiebreak_score(question:str, token:str)->int:
    """Weak heuristic for ties only: prefer tokens compatible with units/percent cues in the question."""
    q = (question or "").lower()
    t = _to_float(token)
    if t is None: return 0
    score = 0
    if "$" in q or "dollar" in q or "usd" in q or "brl" in q:
        # prefer non-fractional money; lightly prefer multiples of 5/10
        if float(t).is_integer(): score += 1
        if abs(round(t/5)*5 - t) < 1e-9: score += 1
        if abs(round(t/10)*10 - t) < 1e-9: score += 1
    if "percent" in q or "%" in q:
        # prefer values within [0,100]
        if 0.0 <= t <= 100.0: score += 1
    if "minutes" in q or "hours" in q or "seconds" in q:
        # prefer non-fractional smallish numbers
        if float(t).is_integer(): score += 1
    return score

def panel_judge_select_token(task:str, inds: List[Individual], jt, jm, seed:int, messages:List[Dict[str,Any]], fused_ind:Optional[Individual]=None)->Tuple[Optional[str], Optional[str], Dict[str,Tuple[int,int]]]:
    buckets = _cluster_by_token(inds)
    # include fused candidate as just another token bucket (does not inflate vote counts unless we add it explicitly)
    if fused_ind is not None:
        ftok = _answer_token_of(fused_ind)
        if ftok:
            buckets.setdefault(ftok, []).append(fused_ind)

    if not buckets: return None, None, {}

    if len(buckets)==1:
        only_tok = next(iter(buckets.keys()))
        return only_tok, "single_token_only", {(only_tok):(1, len(buckets[only_tok]))}

    pr, token_list, support_map = _build_panel_prompt(task, buckets)
    outs, _ = robust_generate_batch(jt, jm, [pr], MAX_NEW_TOKENS_PANEL_JUDGE, None, None, seed, 1, True)  # greedy
    raw = outs[0] if outs else ""
    _append_msg(messages, {"role":"judge","type":"panel_final","prompt":middle_truncate_text(pr,MSG_TRUNC),"output":middle_truncate_text(raw,MSG_TRUNC)})
    m = PANEL_PARSE_RE.search(raw or "")
    why = None
    if m:
        cand = _canon(m.group(1))
        why_m = re.search(r"(?im)^\s*WHY\s*:\s*(.+)$", raw)
        if why_m: why = why_m.group(1).strip()
        if cand and cand in token_list:
            return cand, (why or "panel-picked"), support_map

    # fallback: majority by #cells, then #cands
    if support_map:
        best_tok = sorted(support_map.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))[0][0]
        return best_tok, "fallback_majority_cells", support_map

    # extreme fallback: majority by population
    majority = max(((tok, len(lst)) for tok,lst in buckets.items()), key=lambda t:t[1])[0]
    return majority, "fallback_majority_population", support_map

# ================= Per-task evolution =================
def run_pairwise_generation(task:str, pop:List[Individual], jt, jm, seed:int, messages:List[Dict[str,Any]]):
    return run_pairwise_tournament(task, pop, jt, jm, MATCHES_PER_GEN, seed, LOG_EVERY, messages)

def _choose_backstop_from_elites(champs:List[Individual])->Optional[str]:
    for ind in champs:
        tok = _canon(get_sections(ind.text).get("ANSWER","").strip()) or _canon(_extract_token(ind.text or ""))
        if tok: return ind.text
    return None

def _per_cell_majority(champs:List[Individual])->Optional[str]:
    buckets={}
    for ind in champs:
        key=(ind.tactic,ind.length_bin,ind.checks_bin)
        tok = _canon(get_sections(ind.text).get("ANSWER","").strip()) or _canon(_extract_token(ind.text))
        if not tok: continue
        buckets.setdefault(key,{}); buckets[key][tok]=buckets[key].get(tok,0)+1
    token_cells={}
    for key,counts in buckets.items():
        best = max(counts.items(), key=lambda kv: kv[1])[0]
        token_cells.setdefault(best,set()).add(key)
    for tok,cells in token_cells.items():
        if len(cells)>=2: return tok
    return None

def synthesize_best(at, am, task: str, leaders: List[Individual], seed: int, guidance: Optional[str], messages:List[Dict[str,Any]]):
    top = leaders[:3] if len(leaders) >= 3 else leaders
    pieces = []
    for k, ind in enumerate(top, 1):
        s = get_sections(ind.text)
        pieces.append(
            f"[CAND {k}] <PLAN>\n{s.get('PLAN','')}\n</PLAN>\n"
            f"<SOLVE>\n{s.get('SOLVE','')}\n</SOLVE>\n"
            f"<CHECK>\n{s.get('CHECK','')}\n</CHECK>"
        )
    fragments = "\n\n".join(pieces)
    header = build_author_header_with_guidance(task, guidance if COACH_ENABLED else None)
    prompt = (header +
        "Fuse the best ideas and produce a concise final answer with these tags only:\n"
        "<SETUP>...</SETUP>\n<PLAN>...</PLAN>\n<SOLVE>...</SOLVE>\n<CHECK>...</CHECK>\n<ANSWER>...</ANSWER>\n<META>...</META>\n"
        "Rules: ONLY numeric value in <ANSWER>; no text outside tags.\n\n"
        f"Task:\n{normalize_task(task)}\n\n"
        f"Useful fragments:\n{fragments}\n")
    gens, st = robust_generate_batch(at, am, [prompt], 256, TEMP_AUTHOR, TOPP_AUTHOR, seed, 1, False)
    out = gens[0]
    _append_msg(messages, {"role":"author","type":"fuse","prompt":middle_truncate_text(prompt,MSG_TRUNC),"output":middle_truncate_text(out,MSG_TRUNC)})
    if not is_valid_tagged_answer(out):
        out2, reason = repair_or_synthesize(at, am, task, out, seed+999)
        _append_msg(messages, {"role":"author","type":"repair","reason":reason,"output":middle_truncate_text(out2,MSG_TRUNC)})
        out = out2
    return out, st

def choose_final_answer(task:str, champs:List[Individual], fused_text:str, jt, jm, seed:int, messages:List[Dict[str,Any]], all_inds:List[Individual]) -> Tuple[str, str, Dict[str,Tuple[int,int]], Optional[str]]:
    fused_tok = _canon(get_sections(fused_text).get("ANSWER","").strip()) or _canon(_extract_token(fused_text))
    # create pseudo-individual for fused so it appears in panel, but does not distort cell counts beyond 1 cand
    fused_ind = None
    if fused_tok:
        fused_ind = Individual(id="fused", text=fused_text, tactic="fused", detail=2, target_checks=1,
                               tokens=len(fused_text), checks_found=0, length_bin="medium", checks_bin="low", elo=999.0)

    # === GEJ: global panel judge across ALL individuals (plus fused) ==========
    panel_token, panel_why, support_map = panel_judge_select_token(task, all_inds if all_inds else champs, jt, jm, seed+55, messages, fused_ind=fused_ind if fused_tok else None)
    if panel_token:
        # pick best representative that matches the chosen token
        pool = [x for x in (all_inds if all_inds else champs) if _answer_token_of(x)==panel_token]
        # include fused in selection pool if it matches
        if fused_ind is not None and _answer_token_of(fused_ind)==panel_token:
            pool = pool + [fused_ind]
        rep = max(pool, key=lambda x:x.elo) if pool else (champs[0] if champs else None)
        if rep:
            return rep.text, f"panel_token_select:{panel_why}", support_map, panel_token
    # =========================================================================

    # Legacy backstops: majority per-cell then elites/population; fuser NEVER overrides these
    vote_tok = _per_cell_majority(champs)
    if vote_tok:
        synth = "<SETUP></SETUP>\n<PLAN></PLAN>\n<SOLVE></SOLVE>\n<CHECK></CHECK>\n" \
                f"<ANSWER>{vote_tok}</ANSWER>\n<META>majority-backstop</META>\n"
        return synth, "fused_majority_fix", support_map, vote_tok

    # population majority across all inds
    buckets_all = _cluster_by_token(all_inds if all_inds else champs)
    if buckets_all:
        # tie-break with weak unit heuristic
        by_pop = sorted(buckets_all.items(), key=lambda kv: (-len(kv[1]), -max((x.elo for x in kv[1]), default=0.0), kv[0]))
        tied = [tok for tok, lst in by_pop if len(lst)==len(by_pop[0][1])]
        if len(tied)>1:
            best = sorted(tied, key=lambda tok: (-_unit_tiebreak_score(task, tok), tok))[0]
        else:
            best = by_pop[0][0]
        synth = "<SETUP></SETUP>\n<PLAN></PLAN>\n<SOLVE></SOLVE>\n<CHECK></CHECK>\n" \
                f"<ANSWER>{best}</ANSWER>\n<META>population-majority</META>\n"
        return synth, "fallback_majority_population", support_map, best

    # elite backstop
    fb = _choose_backstop_from_elites(champs)
    if fb: 
        tok_fb = _canon(get_sections(fb).get("ANSWER","").strip()) or _canon(_extract_token(fb))
        return fb, "elite_backstop", support_map, tok_fb

    # nothing worked; return fused (may be empty)
    return fused_text, "last_resort_fused", support_map, fused_tok

def run_evolution_for_task(task_id:str, task_text:str, at, am, jt, jm, messages:List[Dict[str,Any]])->Dict[str,Any]:
    guidance = None
    all_inds: List[Individual] = []
    # Init
    swap_active_models(am, jm, author_on_gpu=True)
    pop=init_population(at, am, task_text, P, SEED_AUTHOR*1000+(hash(task_id)%1000), guidance, messages)
    all_inds.extend(pop)
    archive = MapElites(); archive.consider_all(pop) if USE_PERSISTENT_MAP_ELITES else None
    # Judge init
    swap_active_models(am, jm, author_on_gpu=False)
    tour=run_pairwise_generation(task_text, pop, jt, jm, SEED_JUDGE*1000+(hash(task_id)%1000), messages)
    upset_hist=[tour["upsets"]/max(1,tour["matches"]) if tour["matches"] else 0.0]
    # Coach after init
    if COACH_ENABLED:
        swap_active_models(am, jm, author_on_gpu=False)
        hint, _ = coach_generate_hint(task_text, pop, at, am, jt, jm, SEED_JUDGE*5000+(hash(task_id)%5003), messages)
        if hint: guidance = hint
    # Generations
    for gen in range(1,GENERATIONS):
        elites = (archive.per_cell_elites(ELITISM_KEEP) if (USE_MAP_ELITES_FOR_SELECTION and USE_PERSISTENT_MAP_ELITES)
                  else sorted(pop, key=lambda x:x.elo, reverse=True)[:ELITISM_KEEP])
        mut_r, cross_r = MUT_RATE, (CROSS_RATE if USE_CROSSOVER else 0.0)
        s = max(1e-9, mut_r+cross_r); mut_r, cross_r = mut_r/s, cross_r/s
        target=max(0,P-len(elites)); nmut=int(round(target*mut_r)); ncross=max(0,target-nmut)
        champs=archive.champions() if USE_PERSISTENT_MAP_ELITES else sorted(pop,key=lambda x:x.elo, reverse=True)
        pool=champs if len(champs)>=2 else sorted(pop,key=lambda x:x.elo, reverse=True)
        children=[]
        swap_active_models(am, jm, author_on_gpu=True)
        if nmut>0:
            parents=[random.choice(pool) for _ in range(nmut)]
            ids=[f"gen{gen}_m{k:02d}" for k in range(nmut)]
            newkids = author_mutate_batch(at, am, task_text, parents, ids, SEED_AUTHOR*(10000+gen*100)+(hash(task_id)%997), guidance, messages)
            children+=newkids; all_inds.extend(newkids)
        if ncross>0 and len(pool)>=2:
            As,Bs=[],[]
            for _ in range(ncross):
                a,b=random.sample(pool,2); As.append(a); Bs.append(b)
            ids=[f"gen{gen}_c{k:02d}" for k in range(ncross)]
            newkids = author_crossover_batch(at, am, task_text, As, Bs, ids, SEED_AUTHOR*(20000+gen*100)+(hash(task_id)%991), guidance, messages)
            children+=newkids; all_inds.extend(newkids)
        pop=elites+children
        if len(pop)>P: pop=pop[:P]
        if USE_PERSISTENT_MAP_ELITES: archive.consider_all(pop)
        else: archive.archive = {archive.key(x):x for x in pop}
        swap_active_models(am, jm, author_on_gpu=False)
        tour=run_pairwise_generation(task_text, pop, jt, jm, SEED_JUDGE*(1000+gen)+(hash(task_id)%1000), messages)
        upset=tour["upsets"]/max(1,tour["matches"]) if tour["matches"] else 0.0; upset_hist.append(upset)
        if len(upset_hist)>=3 and abs(upset_hist[-1]-upset_hist[-2])<EARLY_STOP_DELTA and abs(upset_hist[-2]-upset_hist[-3])<EARLY_STOP_DELTA:
            tqdm.write(f"[Early stop] Upset stabilized @ gen {gen}."); break
        # Coach for NEXT generation
        if COACH_ENABLED:
            swap_active_models(am, jm, author_on_gpu=False)
            hint, _ = coach_generate_hint(task_text, pop, at, am, jt, jm, SEED_JUDGE*(7000+gen*77)+(hash(task_id)%7919), messages)
            if hint: guidance = hint
    # Fuse (used as an additional candidate; cannot override consensus)
    swap_active_models(am, jm, author_on_gpu=True)
    champs=archive.champions()
    fused, _ = synthesize_best(at, am, task_text, champs, SEED_AUTHOR*9999+(hash(task_id)%9973), guidance, messages)
    # Choose final (GEJ first)
    swap_active_models(am, jm, author_on_gpu=False)
    chosen_text, final_strategy, support_map, panel_token = choose_final_answer(task_text, champs, fused, jt, jm, SEED_JUDGE*777+(hash(task_id)%997), messages, all_inds)
    return {"final_text": chosen_text, "final_strategy": final_strategy, "support_map": support_map, "panel_token": panel_token}

# ================= Benchmark IO =================
def read_math_benchmark_jsonl(path:str)->List[Tuple[str,str,str]]:
    tasks=[]
    with open(path,"r",encoding="utf-8") as f:
        for idx,line in enumerate(f):
            line=line.strip()
            if not line: continue
            obj=json.loads(line)
            q=obj.get("question","").strip(); a_field=obj.get("answer",""); tid=obj.get("id",f"{idx:05d}")
            tasks.append((str(tid), q, a_field))
    return tasks

# ================= Stats helpers (Wilson CI) =================
def _wilson_ci(success:int, n:int, z:float=1.96)->Tuple[float,float]:
    if n<=0: return (0.0, 0.0)
    phat = success / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    margin = (z*math.sqrt((phat*(1-phat)+z*z/(4*n))/n)) / denom
    lo, hi = max(0.0, center-margin), min(1.0, center+margin)
    return (lo, hi)

# ================= Main =================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(BENCH_PATH): raise FileNotFoundError(f"Benchmark file not found at: {BENCH_PATH}")

    tqdm.write("== Loading AUTHOR model (bf16, on GPU) ...")
    at, am, a_attn = load_llm_bf16(AUTHOR_MODEL_NAME, SEED_AUTHOR, "cuda")
    tqdm.write("== Loading JUDGE model (bf16, on CPU) ...")
    jt, jm, j_attn = load_llm_bf16(JUDGE_MODEL_NAME, SEED_JUDGE, "cpu")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    loader_mode = f"bf16_{a_attn}_{device_str}"
    run_dir = os.path.join(OUT_DIR, f"evo_{AUTHOR_MODEL_NAME.split('/')[-1]}_{loader_mode}")
    os.makedirs(run_dir, exist_ok=True)

    done_ids=set()
    if RESUME_RUN:
        for fn in os.listdir(run_dir):
            if fn.startswith("task_") and fn.endswith(".json"): done_ids.add(fn[5:-5])

    bench = read_math_benchmark_jsonl(BENCH_PATH)
    if not bench: raise RuntimeError("No tasks found in benchmark file.")

    pending = [(tid,q,g) for (tid,q,g) in bench if not (RESUME_RUN and tid in done_ids)]
    if TASK_LIMIT is not None:
        pending = pending[:TASK_LIMIT]

    start=time.time(); total_seen=0; total_time=0.0; total_out_tokens=0
    gold_num_seen=0; num_correct=0
    partials_path=os.path.join(run_dir,"partials.jsonl")

    for tid,question,gold_field_full in pending:
        messages: List[Dict[str,Any]] = []  # per-task message log

        # ===== Evolution (no gold influence) =====
        evo = run_evolution_for_task(tid, question, at, am, jt, jm, messages)
        model_answer_core = evo["final_text"]
        model_final_token = _canon(get_sections(model_answer_core).get("ANSWER","").strip()) or _canon(_extract_token(model_answer_core))

        # Ensure FINAL_ANSWER line
        if model_final_token:
            last = _last_line(model_answer_core)
            ok = bool(our_final_line_pat.search(last) and _canon(our_final_line_pat.search(last).group(1))==model_final_token)
            final_line = f"{FINAL_TAG} {model_final_token}"
            model_answer_raw = model_answer_core if ok else (model_answer_core.rstrip()+("\n" if not model_answer_core.endswith("\n") else "")+final_line)
        else:
            model_answer_raw = model_answer_core.rstrip()+("\n" if not model_answer_core.endswith("\n") else "")+f"{FINAL_TAG} "

        out_tokens = len(at.encode(model_answer_raw, add_special_tokens=False))

        # ===== Parse gold AFTER saving result (for eval only) =====
        gold_tok_raw = (hash4_pat.search(gold_field_full).group(1).strip() if (gold_field_full and hash4_pat.search(gold_field_full)) else _extract_token(gold_field_full or ""))
        gold_final_token = _canon(gold_tok_raw)

        # Save per-task JSON (now includes gold + panel summary)
        support_map = evo.get("support_map", {})
        panel_token = evo.get("panel_token")
        rec = {
            "task_id": tid,
            "question": question,
            "gold_field_raw": gold_field_full,
            "gold_final_token": gold_final_token,
            "model_answer_raw": model_answer_raw,
            "model_final_token": model_final_token,
            "model_name": AUTHOR_MODEL_NAME,
            "loader_mode": loader_mode,
            "max_new_tokens_start": MAX_NEW_TOKENS_AUTHOR_CEIL,
            "batch_size_start": BATCH_AUTHOR,
            "gen_seconds": 0.0,  # fuse seconds omitted for brevity
            "out_tokens": out_tokens,
            "final_strategy": evo.get("final_strategy",""),
            "panel_decision_token": panel_token,
            "panel_support": support_map,  # token -> (cells, cands)
        }
        with open(os.path.join(run_dir, f"task_{tid}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        # Also dump messages json (already includes answer_line fields)
        msg_path = os.path.join(run_dir, f"task_{tid}_messages.json")
        with open(msg_path, "w", encoding="utf-8") as f:
            json.dump({"task_id":tid, "messages":messages}, f, ensure_ascii=False, indent=2)

        # Eval stats (numeric exact)
        total_seen += 1; total_time += 0.0; total_out_tokens += out_tokens
        gnum = _to_float(gold_final_token); pnum = _to_float(model_final_token)
        if (gnum is not None) and (pnum is not None):
            gold_num_seen += 1
            if math.isclose(gnum, pnum, rel_tol=1e-6, abs_tol=1e-8): num_correct += 1

        if total_seen % REPORT_EVERY == 0:
            elapsed = time.time()-start
            num_acc = (num_correct/gold_num_seen) if gold_num_seen else 0.0
            nm_lo,nm_hi = _wilson_ci(num_correct, gold_num_seen) if gold_num_seen else (0.0,0.0)
            avg_tok = total_out_tokens/total_seen if total_seen else 0.0
            tput = total_seen/max(elapsed,1e-9)
            print(f"[Partial @ {total_seen}] Numeric {num_acc*100:.1f}% (95% CI {nm_lo*100:.1f}–{nm_hi*100:.1f}, n={gold_num_seen}); avg out tok {avg_tok:.1f}; throughput {tput:.2f}/s; loader={loader_mode}", flush=True)
            with open(partials_path,"a",encoding="utf-8") as fp:
                fp.write(json.dumps({
                    "seen": total_seen, "gold_seen": gold_num_seen, "num_correct": num_correct,
                    "num_acc": num_acc, "num_ci95": [nm_lo, nm_hi],
                    "avg_out_tokens": avg_tok, "throughput_qps": tput, "model_name": AUTHOR_MODEL_NAME, "loader_mode": loader_mode
                }, ensure_ascii=False)+"\n")

        print(f"[Task {tid}] final <ANSWER>={model_final_token} | gold={gold_final_token} | strategy={evo.get('final_strategy')}")

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_name": AUTHOR_MODEL_NAME, 
            "judge_model_name": JUDGE_MODEL_NAME,
            "device": device_str, "dtype": str(DTYPE),
            "benchmark_path": BENCH_PATH, "total": total_seen, "timing_seconds": total_time, "loader_mode": loader_mode,
            "gold_num_seen": gold_num_seen, "num_correct": num_correct,
            "num_acc": (num_correct/gold_num_seen) if gold_num_seen else None,
            "ablations": {
                "use_persistent_map_elites": USE_PERSISTENT_MAP_ELITES,
                "use_map_elites_for_selection": USE_MAP_ELITES_FOR_SELECTION,
                "use_crossover": USE_CROSSOVER,
                "enable_cost_normalization": ENABLE_COST_NORMALIZATION,
                "cost_norm_lambda": COST_NORM_LAMBDA,
                "cost_norm_tau": COST_NORM_TAU,
                "final_judge_on_fused": FINAL_JUDGE_ON_FUSED,
                "coach_enabled": COACH_ENABLED,
                "coach_topk_children": COACH_TOPK_CHILDREN,
                "use_panel_final_selection": USE_PANEL_FINAL_SELECTION,
                "panel_topk_per_token": PANEL_TOPK_PER_TOKEN,
                "panel_max_tokens": PANEL_MAX_TOKENS
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSaved per-task outputs & messages in: {run_dir}")

if __name__ == "__main__":
    main()
