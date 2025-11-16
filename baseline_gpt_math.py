# -*- coding: utf-8 -*-
"""
GPT-OSS-20B simple baseline — crash-safe, OOM-safe, clear FINAL_ANSWER line (no CoT prompt)
Usage: python baseline_oomsafe_progress.py

Fixes:
- Robust FINAL_ANSWER parsing even when the model adds stray prefixes (e.g., "assistantfinalFINAL_ANSWER: 3").
- Sanitizes any trailing FINAL_ANSWER lines (empty or duplicated) and appends exactly one clean line: "FINAL_ANSWER: <value>".
- Stopping criteria now detect FINAL_ANSWER anywhere on the line (not only at column 0).

Notes:
- Exact Match (EM) against full gold text is removed; we only care about the final numeric answer.
- Metrics focus on numeric correctness (num_correct / gold_num_seen), plus throughput stats.
- Loader prioritizes 4-bit quantization for 20B viability; falls back gracefully.
"""

import os, json, time, gc, math, re, ast
from typing import List, Tuple, Optional, Dict, Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

# ===================== Config =====================

MODEL_NAME   = "openai/gpt-oss-20b"
BENCH_PATH   = "/scratch/pedro.bento/evostar/data/math_benchmark.jsonl"
OUT_DIR      = "gptoss20b_runs"

MAX_NEW_TOKENS = 256    # 20B-friendly cap; early stop keeps it fast
BATCH_SIZE     = 1      # 20B-friendly default
REPORT_EVERY   = 10
RESUME_RUN     = True

# Limit how many tasks to run this session (set to None for all)
TASK_LIMIT     = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

OFFLOAD_DIR = os.path.join(OUT_DIR, "offload_cache")
os.makedirs(OFFLOAD_DIR, exist_ok=True)

FINAL_TAG = "FINAL_ANSWER:"

# ===================== IO =====================

def read_math_benchmark_jsonl(path: str) -> List[Tuple[str, str, Optional[str]]]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not (line := line.strip()): continue
            o = json.loads(line)
            tasks.append((str(o.get("id", f"{idx:05d}")), o.get("question","").strip(), o.get("answer")))
    return tasks

# ===================== Loader (OOM-safe, 20B-first quantization) =====================

def _try_full_gpu(name, dtype):
    m = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to("cuda")
    return m, "full_gpu_bf16"

def _try_auto_offload(name, dtype):
    m = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
        offload_folder=OFFLOAD_DIR, offload_state_dict=True, low_cpu_mem_usage=True
    )
    return m, "auto_offload_bf16"

def _try_8bit(name):
    if BitsAndBytesConfig is None: raise RuntimeError("bitsandbytes unavailable")
    q = BitsAndBytesConfig(load_in_8bit=True)
    m = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, quantization_config=q, device_map="auto",
        offload_folder=OFFLOAD_DIR, low_cpu_mem_usage=True
    )
    return m, "8bit_auto"

def _try_4bit(name):
    if BitsAndBytesConfig is None: raise RuntimeError("bitsandbytes unavailable")
    q = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)
    )
    m = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, quantization_config=q, device_map="auto",
        offload_folder=OFFLOAD_DIR, low_cpu_mem_usage=True
    )
    return m, "4bit_auto_nf4"

def load_model(name: str):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    torch.set_float32_matmul_precision("high")

    last_err = None
    # For 20B, prefer quantized/offload first
    for loader in (_try_4bit, _try_8bit, _try_auto_offload, _try_full_gpu):
        try:
            if loader in (_try_full_gpu, _try_auto_offload):
                mdl, mode = loader(name, DTYPE)
            else:
                mdl, mode = loader(name)
            break
        except Exception as e:
            last_err = e
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        raise RuntimeError(f"Model load failed. Last error: {last_err}")

    mdl.config.use_cache = True
    if getattr(mdl, "generation_config", None) is not None:
        mdl.generation_config.pad_token_id = tok.eos_token_id
    mdl.eval()
    print(f"[Loader] Using mode: {mode}")
    return tok, mdl, mode

# ===================== Prompt & Stop =====================

def build_user_prompt(q: str) -> str:
    # No CoT instruction. Just a machine-readable final line.
    return (
        f"Solve the problem.\n\nProblem:\n{q}\n\n"
        f"At the end, output exactly one line:\n{FINAL_TAG} <value>\n"
        f"Do not output anything after that line."
    )

def apply_chat_template(tokenizer, user_text: str) -> str:
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                [{"role": "user","content": user_text}],
                add_generation_prompt=True, tokenize=False
            )
    except Exception:
        pass
    return user_text

class FinalAnswerStopper(StoppingCriteria):
    """
    Stop when a completed 'FINAL_ANSWER: <value>' appears anywhere on a line
    (some models prefix with garbage like 'assistantfinal').
    """
    def __init__(self, tokenizer, start_len: int, eos_id: Optional[int], lookback_tokens: int = 256):
        super().__init__()
        self.tok = tokenizer
        self.start_len = start_len
        self.eos_id = eos_id
        self.n = lookback_tokens
        # allow preceding junk and require a non-empty value
        self.done_nl  = re.compile(r"(?mi).*FINAL_ANSWER\s*:\s*\S[^\n\r]*\r?\n")
        self.done_eos = re.compile(r"(?mi).*FINAL_ANSWER\s*:\s*\S[^\n\r]*\s*$")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen = input_ids[0][self.start_len:]
        if gen.numel() == 0:
            return False
        tail = gen[-self.n:] if gen.size(0) > self.n else gen
        tail_txt = self.tok.decode(tail, skip_special_tokens=True)
        if self.done_nl.search(tail_txt):
            return True
        if (self.eos_id is not None) and (gen[-1].item() == self.eos_id):
            full = self.tok.decode(gen, skip_special_tokens=True)
            if self.done_eos.search(full):
                return True
        return False

def _primary_device(model) -> torch.device:
    try: return next(model.parameters()).device
    except StopIteration: return torch.device("cpu")

# ===================== Generation =====================

def stream_generate(tokenizer, model, questions: List[str], max_new_tokens: int, batch_size: int):
    """Yield: (index, output_text, gen_seconds, out_token_count) — with early stop on FINAL_ANSWER."""
    dev = _primary_device(model)
    micro_bs = max(1, min(batch_size, len(questions)))
    max_new = max_new_tokens
    i = 0
    with torch.inference_mode():
        while i < len(questions):
            qs = questions[i:i+micro_bs]
            prompts = [apply_chat_template(tokenizer, build_user_prompt(q)) for q in qs]
            try:
                t0 = time.time()
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
                inputs = {k: v.to(dev) for k, v in inputs.items()}
                start_len = inputs["input_ids"].shape[1]
                stopper = StoppingCriteriaList([FinalAnswerStopper(tokenizer, start_len, tokenizer.eos_token_id, 256)])
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopper
                )
                dt = time.time() - t0
                dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
                for j, full in enumerate(dec):
                    pref = prompts[j]
                    out = full[len(pref):].strip() if full.startswith(pref) else full.strip()
                    out_tok = len(tokenizer(out, return_tensors=None).input_ids)
                    yield (i + j, out, dt / len(dec), out_tok)
                i += micro_bs
                del inputs, gen
                if torch.cuda.is_available(): torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                if micro_bs > 1:
                    micro_bs = max(1, micro_bs // 2); print(f"[Gen OOM] micro-batch -> {micro_bs}"); continue
                elif max_new > 64:
                    max_new = max(64, max_new // 2); print(f"[Gen OOM] max_new_tokens -> {max_new}"); continue
                else:
                    raise

# ===================== Extraction (robust, compact) =====================

_ws = re.compile(r"\s+")
num_pat  = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
frac_pat = re.compile(r"(?:\\frac\{([^}]*)\}\{([^}]*)\})|(\d+)\s*/\s*(\d+)")
hash4_pat = re.compile(r"####\s*([^\n\r]+)")
th_en = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
th_eu = re.compile(r"[-+]?\d{1,3}(?:\.\d{3})+(?:,\d+)?")
dec_comma = re.compile(r"[-+]?\d+,\d+")
all_zero = re.compile(r"^[+-]?0+(?:[.,]0+)?$")

# NEW: accept FINAL_ANSWER anywhere on a line, require non-empty value
our_final_line_pat = re.compile(r"(?i)FINAL_ANSWER\s*:\s*(\S[^\n\r]*)")
final_line_pat     = re.compile(r"(?im)^\s*(?:final\s*answer|answer|result)\s*[:=]\s*(.+?)\s*$")
rhetorical_pat     = re.compile(r"(?i)(?:thus|therefore|so|hence|consequently)[, ]+(?:the )?answer(?: is)?\s*[:=]?\s*([^\n\r]+)")
mixed_frac         = re.compile(r"^\s*([+-]?\d+)\s+(\d+)\s*/\s*(\d+)\s*$")

def _strip_latex(s: str) -> str: return s.replace("$","").replace("\\(","").replace("\\)","")
def _normalize_text(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    return _ws.sub(" ", _strip_latex(s.replace("\u200b","").strip())).lower().strip(" .")
def _boxed_all(s: str) -> List[str]: return re.findall(r"\\boxed\{([^}]*)\}", s)

def _norm_num(tok: str) -> str:
    t = tok.strip()
    # Trim common surrounding quotes/brackets occasionally emitted by models
    if len(t) >= 2 and ((t[0], t[-1]) in {("<",">"), ("«","»"), ("“","”"), ("\"","\""), ("'","'"), ("`","`")}):
        t = t[1:-1].strip()
    # "a b/c" -> improper fraction
    m = mixed_frac.fullmatch(t)
    if m:
        a,b,c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sign = -1 if a<0 else 1; a = abs(a); num = sign*(a*c+b); return f"{num}/{c}"
    m = re.fullmatch(r"\s*([^\s/]+)\s*/\s*([^\s/]+)\s*", t)
    if m: return f"{m.group(1).strip()}/{m.group(2).strip()}"
    t = t.replace("%","")
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".") if (t.rfind(",")>t.rfind(".")) else t.replace(",", "")
    elif "," in t:
        t = t.replace(",", ".") if dec_comma.fullmatch(t) else t.replace(",", "")
    t = t.replace(" ","").replace("_","")
    for sym in ["$","€","£","R$","USD","BRL"]: t = t.replace(sym,"")
    if re.search(r"[A-Za-z]$", t): t = re.sub(r"([^\W\d_]+)$","",t)
    t = re.sub(r"\s*×\s*10\^([+-]?\d+)", r"e\1", t)
    return t.strip()

def _canon(token: Optional[str]) -> Optional[str]:
    if token is None: return None
    t = token.strip()
    if not t: return None
    m = re.fullmatch(r"\s*([^\s/]+)\s*/\s*([^\s/]+)\s*", t)
    return (f"{_norm_num(m.group(1))}/{_norm_num(m.group(2))}" if m else _norm_num(t))

def _find_last_number(s: str) -> Optional[str]:
    s0 = _strip_latex(s); matches=[]
    for pat in (th_en, th_eu):
        matches += [(m.group(0).strip(), m.end()) for m in pat.finditer(s0) if not all_zero.fullmatch(m.group(0).strip())]
    matches += [(m.group(0).strip(), m.end()) for m in dec_comma.finditer(s0) if not all_zero.fullmatch(m.group(0).strip())]
    matches += [(m.group(0).strip(), m.end()) for m in num_pat.finditer(s0) if not all_zero.fullmatch(m.group(0).strip())]
    if not matches: return None
    return max(matches, key=lambda x:x[1])[0]

def _extract_token(s: str) -> Optional[str]:
    if not s: return None
    # our explicit tag (anywhere in line)
    m = our_final_line_pat.findall(s)
    if m:
        # prefer the last non-empty capture
        return _norm_num(m[-1].strip())
    # fallbacks:
    boxes = _boxed_all(s)
    if boxes:
        b = boxes[-1]; mf = frac_pat.search(b)
        if mf:
            num, den = (mf.group(1), mf.group(2)) if mf.group(1) else (mf.group(3), mf.group(4))
            return f"{_norm_num(num)}/{_norm_num(den)}"
        t = _find_last_number(b)
        if t: return _norm_num(t)
        b = _ws.sub(" ", _strip_latex(b)).strip()
        if b: return b
    m = hash4_pat.search(s)
    if m: return _norm_num(m.group(1).strip())
    m = final_line_pat.findall(s)
    if m: return _norm_num(m[-1])
    m = list(rhetorical_pat.finditer(s))
    if m: return _norm_num(m[-1].group(1))
    mf = frac_pat.search(s)
    if mf:
        num, den = (mf.group(1), mf.group(2)) if mf.group(1) else (mf.group(3), mf.group(4))
        return f"{_norm_num(num)}/{_norm_num(den)}"
    t = _find_last_number(s)
    return _norm_num(t) if t else None

def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip(): return line.strip()
    return ""

def _strip_trailing_final_lines(text: str) -> str:
    """Remove any trailing lines that contain 'FINAL_ANSWER:' (empty or not)."""
    lines = text.splitlines()
    while lines and re.search(r"(?i)final_answer\s*:", lines[-1]):
        lines.pop()
    return "\n".join(lines)

def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n==0: return (0.0,0.0)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); m=z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (max(0.0,(c-m)/d), min(1.0,(c+m)/d))

# ===================== Main =====================

def main():
    if not os.path.exists(BENCH_PATH): raise FileNotFoundError(BENCH_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"== Loading: {MODEL_NAME} (device={DEVICE}, dtype={DTYPE})")
    tok, mdl, mode = load_model(MODEL_NAME)

    tasks = read_math_benchmark_jsonl(BENCH_PATH)
    if not tasks: raise RuntimeError("No tasks found.")
    model_tag = MODEL_NAME.replace("/", "_")
    run_dir = os.path.join(OUT_DIR, f"{model_tag}_{mode}")
    os.makedirs(run_dir, exist_ok=True)

    done = {fn[len("task_"):-5] for fn in os.listdir(run_dir)
            if fn.startswith("task_") and fn.endswith(".json")} if RESUME_RUN else set()

    # Build pending list and apply TASK_LIMIT
    pending = [(tid, q, a) for (tid, q, a) in tasks if not (RESUME_RUN and tid in done)]
    if TASK_LIMIT is not None:
        pending = pending[:TASK_LIMIT]

    start = time.time(); seen=0; ttime=0.0; outtok=0
    gold_num_seen=num_ok=0
    partials_path = os.path.join(run_dir, "partials.jsonl")

    print(f"Generating {len(pending)} answers... report every {REPORT_EVERY}")
    for tid, q, gold_raw in pending:
        sgen = stream_generate(tok, mdl, [q], MAX_NEW_TOKENS, BATCH_SIZE)
        _idx, ans, dt, out_tok = next(sgen)

        # Parse the model token robustly
        model_tok = _canon(_extract_token(ans))

        # Always sanitize tail and append exactly one clean FINAL_ANSWER line
        sanitized_body = _strip_trailing_final_lines(ans).rstrip()
        suffix = f"{FINAL_TAG} {model_tok}" if model_tok else f"{FINAL_TAG} "
        ans_marked = (sanitized_body + ("\n" if sanitized_body else "")) + suffix

        # Parse gold
        gold_tok_raw = (hash4_pat.search(gold_raw).group(1).strip()
                        if (gold_raw and hash4_pat.search(gold_raw))
                        else _extract_token(gold_raw or ""))
        gold_tok = _canon(gold_tok_raw)

        rec = {
            "task_id": tid, "question": q, "gold_field_raw": gold_raw,
            "model_answer_raw": ans_marked,
            "gold_final_token": gold_tok, "model_final_token": model_tok,
            "model_name": MODEL_NAME, "loader_mode": mode,
            "max_new_tokens_start": MAX_NEW_TOKENS, "batch_size_start": BATCH_SIZE,
            "gen_seconds": dt, "out_tokens": out_tok,
        }
        with open(os.path.join(run_dir, f"task_{tid}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        seen += 1; ttime += dt; outtok += out_tok

        # Numeric comparison
        def _to_float(tok: Optional[str]) -> Optional[float]:
            if tok is None: return None
            t = _norm_num(tok)
            try: return float(t)
            except: return None

        def _num_from_text(s: Optional[str]) -> Optional[float]:
            if not s: return None
            mf = frac_pat.search(s)
            if mf:
                a,b = (mf.group(1), mf.group(2)) if mf.group(1) else (mf.group(3), mf.group(4))
                try:
                    return float(_norm_num(a)) / float(_norm_num(b))
                except: pass
            n = _find_last_number(s)
            try: return float(_norm_num(n)) if n else None
            except: return None

        gnum = _to_float(gold_tok) or _num_from_text(gold_raw)
        pnum = _to_float(model_tok) or _num_from_text(ans)
        if (gnum is not None) and (pnum is not None):
            gold_num_seen += 1
            if math.isclose(gnum, pnum, rel_tol=1e-6, abs_tol=1e-8): num_ok += 1

        if seen % REPORT_EVERY == 0:
            el = time.time()-start
            num_acc = (num_ok/gold_num_seen) if gold_num_seen else 0.0
            nm_lo, nm_hi = _wilson_ci(num_ok, gold_num_seen) if gold_num_seen else (0.0,0.0)
            print(f"[{seen}] Num {num_acc*100:.1f}% (n={gold_num_seen}, 95% CI {nm_lo*100:.1f}–{nm_hi*100:.1f}); "
                  f"avg out tok {outtok/seen:.1f}; qps {seen/max(el,1e-9):.2f}; mode={mode}")
            with open(partials_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps({
                    "seen": seen,
                    "gold_num_seen": gold_num_seen,
                    "num_correct": num_ok,
                    "num_acc": num_acc,
                    "num_ci95": [nm_lo, nm_hi],
                    "avg_out_tokens": outtok/seen,
                    "throughput_qps": seen/max(el,1e-9),
                    "elapsed_seconds": el,
                    "model_name": MODEL_NAME,
                    "loader_mode": mode
                }, ensure_ascii=False) + "\n")

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "device": DEVICE,
            "dtype": str(DTYPE),
            "benchmark_path": BENCH_PATH,
            "total": seen,
            "timing_seconds": ttime,
            "loader_mode": mode,
            "gold_num_seen": gold_num_seen,
            "num_correct": num_ok
        }, f, ensure_ascii=False, indent=2)

    del mdl, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"Saved per-task outputs in: {run_dir}\nPartials: {partials_path}")

if __name__ == "__main__":
    main()
