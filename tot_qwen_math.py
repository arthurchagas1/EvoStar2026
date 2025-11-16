# -*- coding: utf-8 -*-
"""
Usage: python baseline_qwen_tot_math.py

Focus:
- Numeric correctness + throughput.
- Robust loader: download full snapshot to a local folder, load locally.
- If shards missing -> force re-download and retry once.
- Tree-of-Thoughts (ToT) style search for better reasoning:
  * Propose: grow THOUGHTS with one small step.
  * Value: score partial THOUGHTS (1–10).
  * Select: keep best states (beam search) across depths.
  * Rollout: from final states, complete reasoning and produce FINAL_ANSWER: <value>.
- If ToT fails, fall back to the original single-shot baseline.
"""

import os, json, time, gc, math, re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

# ===================== Config =====================

MODEL_NAME   = "Qwen/Qwen2.5-7B-Instruct"  # bigger than 7B, instruction-tuned
BENCH_PATH   = "/scratch/pedro.bento/evostar/data/math_benchmark.jsonl"
OUT_DIR      = "tot_qwen_runs_7b_math"

MAX_NEW_TOKENS_BASELINE = 384  # for the fallback single-shot baseline
BATCH_SIZE     = 8
REPORT_EVERY   = 10
RESUME_RUN     = True
TASK_LIMIT     = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

OFFLOAD_DIR   = os.path.join(OUT_DIR, "offload_cache")
LOCAL_MODELS  = os.path.join(OUT_DIR, "local_models")
os.makedirs(OFFLOAD_DIR, exist_ok=True)
os.makedirs(LOCAL_MODELS, exist_ok=True)

FINAL_TAG = "FINAL_ANSWER:"
FINAL_TAG_RE = re.compile(r"(?im)^\s*FINAL_ANSWER\s*:\s*(.+?)\s*$")

# ---- ToT configuration (math) ----

@dataclass
class ToTConfig:
    max_depth: int = 2            # expansion levels
    branch_factor: int = 2        # children per state
    beam_size: int = 3            # states kept per depth
    n_rollouts: int = 2           # rollouts per final state

    max_new_tokens_thought: int = 96
    max_new_tokens_value: int = 64
    max_new_tokens_rollout: int = 192

    temperature_thought: float = 0.7
    temperature_value: float = 0.0   # deterministic scoring
    temperature_rollout: float = 0.7

    top_p_thought: float = 0.9
    top_p_rollout: float = 0.9

TOT_CONFIG = ToTConfig()

# ===================== IO =====================

def read_math_benchmark_jsonl(path: str) -> List[Tuple[str, str, Optional[str]]]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not (line := line.strip()):
                continue
            o = json.loads(line)
            tasks.append((str(o.get("id", f"{idx:05d}")), o.get("question", "").strip(), o.get("answer")))
    return tasks

# ===================== Loader (cache-robust + OOM fallbacks) =====================

def _sanitize_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def _local_repo_path(model_id: str) -> str:
    return os.path.join(LOCAL_MODELS, _sanitize_tag(model_id))

def _snapshot(model_id: str, force: bool = False) -> str:
    """
    Download a complete local snapshot (no filtering) so all shards are present.
    """
    local_dir = _local_repo_path(model_id)
    if snapshot_download is None:
        return model_id  # fallback to hub path (not recommended)
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=None,   # ignored / not used by new hub, safe to pass None
        force_download=force,
    )
    return local_dir

def _tok_from_local(local_path: str):
    tok = AutoTokenizer.from_pretrained(
        local_path, trust_remote_code=True, local_files_only=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def _mdl_full_gpu(local_path: str, dtype):
    m = AutoModelForCausalLM.from_pretrained(
        local_path, trust_remote_code=True, use_safetensors=True,
        low_cpu_mem_usage=True, dtype=dtype, local_files_only=True
    ).to("cuda")
    return m, "full_gpu_bf16" if dtype == torch.bfloat16 else "full_gpu_fp32"

def _mdl_auto_offload(local_path: str, dtype):
    m = AutoModelForCausalLM.from_pretrained(
        local_path, trust_remote_code=True, use_safetensors=True,
        low_cpu_mem_usage=True, dtype=dtype, device_map="auto",
        offload_folder=OFFLOAD_DIR, offload_state_dict=True,
        local_files_only=True
    )
    return m, "auto_offload_bf16" if dtype == torch.bfloat16 else "auto_offload_fp32"

def _mdl_8bit(local_path: str):
    if BitsAndBytesConfig is None:
        raise RuntimeError("bitsandbytes unavailable")
    q = BitsAndBytesConfig(load_in_8bit=True)
    m = AutoModelForCausalLM.from_pretrained(
        local_path, trust_remote_code=True, use_safetensors=True,
        low_cpu_mem_usage=True, quantization_config=q, device_map="auto",
        offload_folder=OFFLOAD_DIR, local_files_only=True
    )
    return m, "8bit_auto"

def _mdl_4bit(local_path: str):
    if BitsAndBytesConfig is None:
        raise RuntimeError("bitsandbytes unavailable")
    q = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)
    )
    m = AutoModelForCausalLM.from_pretrained(
        local_path, trust_remote_code=True, use_safetensors=True,
        low_cpu_mem_usage=True, quantization_config=q, device_map="auto",
        offload_folder=OFFLOAD_DIR, local_files_only=True
    )
    return m, "4bit_auto_nf4"

def _missing_file_err(e: Exception) -> bool:
    msg = str(e).lower()
    return ("no such file or directory" in msg) or ("does not appear to have files named" in msg)

def load_model(name: str):
    # 1) Download full snapshot locally
    local_path = _snapshot(name, force=False)

    # 2) Tokenizer
    tok = _tok_from_local(local_path)

    # 3) Try load paths (OOM-safe). If missing shards, force re-download once and retry.
    torch.set_float32_matmul_precision("high")

    def _try_all(local_path_inner: str):
        last_err = None
        for ctor in (_mdl_full_gpu, _mdl_auto_offload):
            try:
                mdl, mode = ctor(local_path_inner, DTYPE)
                return mdl, mode
            except Exception as e:
                last_err = e
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        for ctor in (_mdl_8bit, _mdl_4bit):
            try:
                mdl, mode = ctor(local_path_inner)
                return mdl, mode
            except Exception as e:
                last_err = e
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        raise last_err

    try:
        mdl, mode = _try_all(local_path)
    except Exception as e1:
        if _missing_file_err(e1):
            print("[Loader] Missing shard(s) detected. Forcing a fresh snapshot and retrying...")
            local_path = _snapshot(name, force=True)
            mdl, mode = _try_all(local_path)
        else:
            raise RuntimeError(f"Model load failed. Last error: {e1}")

    mdl.config.use_cache = True
    if getattr(mdl, "generation_config", None) is not None:
        mdl.generation_config.pad_token_id = tok.eos_token_id
    mdl.eval()
    print(f"[Loader] Using mode: {mode}")
    return tok, mdl, mode

# ===================== Prompt & Stop =====================

def build_user_prompt(q: str) -> str:
    return (
        f"Solve the problem.\n\nProblem:\n{q}\n\n"
        f"At the end, output exactly one line:\n{FINAL_TAG} <value>\n"
        f"Do not output anything after that line."
    )

def apply_chat_template(tokenizer, user_text: str) -> str:
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                add_generation_prompt=True, tokenize=False
            )
    except Exception:
        pass
    return user_text

class FinalAnswerStopper(StoppingCriteria):
    """Stop when a completed 'FINAL_ANSWER: <value>' line appears in generated text (ignoring prompt)."""
    def __init__(self, tokenizer, start_len: int, eos_id: Optional[int], lookback_tokens: int = 256):
        super().__init__()
        self.tok = tokenizer
        self.start_len = start_len
        self.eos_id = eos_id
        self.n = lookback_tokens
        self.done_nl = re.compile(r"(?m)^\s*FINAL_ANSWER\s*:\s*\S[^\n\r]*\r?\n")
        self.done_eos = re.compile(r"(?m)^\s*FINAL_ANSWER\s*:\s*\S[^\n\r]*\s*$")

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
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

# ===================== Generic generation helper (for ToT & others) =====================

def generate_one(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    stop_on_final_answer: bool = False,
    lookback_tokens: int = 256,
) -> Tuple[str, float, int]:
    """
    Single-prompt generation, with simple OOM backoff.

    Returns:
        out_text, gen_seconds, out_token_count
    """
    dev = _primary_device(model)
    cur_max_new = max_new_tokens

    while True:
        try:
            t0 = time.time()
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            start_len = inputs["input_ids"].shape[1]

            stopping = None
            if stop_on_final_answer:
                stopping = StoppingCriteriaList([
                    FinalAnswerStopper(tokenizer, start_len, tokenizer.eos_token_id, lookback_tokens)
                ])

            gen = model.generate(
                **inputs,
                max_new_tokens=cur_max_new,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping,
            )
            dt = time.time() - t0
            full = tokenizer.decode(gen[0], skip_special_tokens=True)
            if full.startswith(prompt):
                out = full[len(prompt):].strip()
            else:
                out = full.strip()

            out_tok = len(tokenizer(out, return_tensors=None).input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return out, dt, out_tok

        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if cur_max_new > 64:
                cur_max_new = max(64, cur_max_new // 2)
                print(f"[Gen OOM] max_new_tokens -> {cur_max_new}")
                continue
            else:
                raise

# ===================== Original streaming baseline (used only as fallback) =====================

def stream_generate(tokenizer, model, questions: List[str], max_new_tokens: int, batch_size: int):
    """Yield: (index, output_text, gen_seconds, out_token_count) — early stop on FINAL_ANSWER."""
    dev = _primary_device(model)
    micro_bs = max(1, min(batch_size, len(questions)))
    max_new = max_new_tokens
    i = 0
    with torch.inference_mode():
        while i < len(questions):
            qs = questions[i:i + micro_bs]
            prompts = [apply_chat_template(tokenizer, build_user_prompt(q)) for q in qs]
            try:
                t0 = time.time()
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
                inputs = {k: v.to(dev) for k, v in inputs.items()}
                start_len = inputs["input_ids"].shape[1]
                stopper = StoppingCriteriaList([FinalAnswerStopper(tokenizer, start_len, tokenizer.eos_token_id, 256)])
                gen = model.generate(
                    **inputs, max_new_tokens=max_new, do_sample=False, top_p=1.0,
                    pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopper
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
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if micro_bs > 1:
                    micro_bs = max(1, micro_bs // 2)
                    print(f"[Gen OOM] micro-batch -> {micro_bs}")
                    continue
                elif max_new > 64:
                    max_new = max(64, max_new // 2)
                    print(f"[Gen OOM] max_new_tokens -> {max_new}")
                    continue
                else:
                    raise

# ===================== Extraction utils =====================

_ws = re.compile(r"\s+")
num_pat  = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
frac_pat = re.compile(r"(?:\\frac\{([^}]*)\}\{([^}]*)\})|(\d+)\s*/\s*(\d+)")
hash4_pat = re.compile(r"####\s*([^\n\r]+)")
th_en = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
th_eu = re.compile(r"[-+]?\d{1,3}(?:\.\d{3})+(?:,\d+)?")
dec_comma = re.compile(r"[-+]?\d+,\d+")
all_zero = re.compile(r"^[+-]?0+(?:[.,]0+)?$")
our_final_line_pat = FINAL_TAG_RE
final_line_pat = re.compile(r"(?im)^\s*(?:final\s*answer|answer|result)\s*[:=]\s*(.+?)\s*$")
rhetorical_pat = re.compile(r"(?i)(?:thus|therefore|so|hence|consequently)[, ]+(?:the )?answer(?: is)?\s*[:=]?\s*([^\n\r]+)")
mixed_frac = re.compile(r"^\s*([+-]?\d+)\s+(\d+)\s*/\s*(\d+)\s*$")

def _strip_latex(s: str) -> str:
    return s.replace("$", "").replace("\\(", "").replace("\\)", "")

def _boxed_all(s: str) -> list:
    return re.findall(r"\\boxed\{([^}]*)\}", s)

def _norm_num(tok: str) -> str:
    t = tok.strip()
    m = mixed_frac.fullmatch(t)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sign = -1 if a < 0 else 1
        a = abs(a)
        num = sign * (a * c + b)
        return f"{num}/{c}"
    m = re.fullmatch(r"\s*([^\s/]+)\s*/\s*([^\s/]+)\s*", t)
    if m:
        return f"{m.group(1).strip()}/{m.group(2).strip()}"
    t = t.replace("%", "")
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".") if (t.rfind(",") > t.rfind(".")) else t.replace(",", "")
    elif "," in t:
        t = t.replace(",", ".") if dec_comma.fullmatch(t) else t.replace(",", "")
    t = t.replace(" ", "").replace("_", "")
    for sym in ["$", "€", "£", "R$", "USD", "BRL"]:
        t = t.replace(sym, "")
    if re.search(r"[A-Za-z]$", t):
        t = re.sub(r"([^\W\d_]+)$", "", t)
    t = re.sub(r"\s*×\s*10\^([+-]?\d+)", r"e\1", t)
    return t.strip()

def _canon(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    t = token.strip()
    if not t:
        return None
    m = re.fullmatch(r"\s*([^\s/]+)\s*/\s*([^\s/]+)\s*", t)
    return (f"{_norm_num(m.group(1))}/{_norm_num(m.group(2))}" if m else _norm_num(t))

def _find_last_number(s: str) -> Optional[str]:
    s0 = _strip_latex(s)
    matches = []
    for pat in (th_en, th_eu):
        matches += [(m.group(0).strip(), m.end()) for m in pat.finditer(s0) if not all_zero.fullmatch(m.group(0).strip())]
    matches += [(m.group(0).strip(), m.end()) for m in dec_comma.finditer(s0) if not all_zero.fullmatch(m.group(0).strip())]
    matches += [(m.group(0).strip(), m.end()) for m in num_pat.finditer(s0) if not all_zero.fullmatch(m.group(0).strip())]
    if not matches:
        return None
    return max(matches, key=lambda x: x[1])[0]

def _extract_token(s: str) -> Optional[str]:
    if not s:
        return None
    m = our_final_line_pat.findall(s)
    if m:
        return _norm_num(m[-1])
    boxes = _boxed_all(s)
    if boxes:
        b = boxes[-1]
        mf = frac_pat.search(b)
        if mf:
            num, den = (mf.group(1), mf.group(2)) if mf.group(1) else (mf.group(3), mf.group(4))
            return f"{_norm_num(num)}/{_norm_num(den)}"
        t = _find_last_number(b)
        if t:
            return _norm_num(t)
        b = _ws.sub(" ", _strip_latex(b)).strip()
        if b:
            return b
    m = hash4_pat.search(s)
    if m:
        return _norm_num(m.group(1).strip())
    m = final_line_pat.findall(s)
    if m:
        return _norm_num(m[-1])
    m = list(rhetorical_pat.finditer(s))
    if m:
        return _norm_num(m[-1].group(1))
    mf = frac_pat.search(s)
    if mf:
        num, den = (mf.group(1), mf.group(2)) if mf.group(1) else (mf.group(3), mf.group(4))
        return f"{_norm_num(num)}/{_norm_num(den)}"
    t = _find_last_number(s)
    return _norm_num(t) if t else None

def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""

def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))

# ===================== ToT (Tree-of-Thoughts) for math =====================

@dataclass
class ThoughtState:
    thoughts: str
    value: float

_score_re = re.compile(r"(?i)SCORE\s*:\s*([0-9]+)")

def _parse_score(text: str) -> Optional[int]:
    m = _score_re.search(text or "")
    if not m:
        return None
    try:
        v = int(m.group(1))
        if 1 <= v <= 10:
            return v
    except Exception:
        return None
    return None

def _tot_propose_prompt_math(question: str, thoughts: str) -> str:
    base = (
        "You are solving a math problem that requires a numerical answer.\n\n"
        f"Problem:\n{question}\n\n"
        "We maintain a running reasoning scratchpad called THOUGHTS.\n"
        "THOUGHTS so far:\n"
    )
    body = thoughts.strip() if thoughts.strip() else "(none yet)"
    tail = (
        "\n\nWrite ONE short new step of reasoning that logically continues from THOUGHTS.\n"
        "- Focus only on analysis; do not restate the problem.\n"
        "- Do NOT compute the final numerical answer yet.\n"
        "- End with a line that contains only the word CONTINUE.\n\n"
        "New step:\n"
    )
    return base + body + tail

def _tot_value_prompt_math(question: str, thoughts: str) -> str:
    return (
        "You are evaluating a partial line of reasoning for a math problem.\n\n"
        f"Problem:\n{question}\n\n"
        "Reasoning so far (THOUGHTS):\n"
        f"{thoughts.strip() or '(none)'}\n\n"
        "Based ONLY on how logically correct, relevant, and helpful this reasoning is for solving the problem,\n"
        "rate it on a scale from 1 to 10.\n"
        "- 1 = clearly flawed or irrelevant\n"
        "- 10 = highly accurate and almost certainly leads to the correct answer\n\n"
        "Reply with exactly one line in the format:\n"
        "SCORE: <integer from 1 to 10>\n"
    )

def _tot_rollout_prompt_math(question: str, thoughts: str) -> str:
    return (
        "You are solving the following math problem. You must produce a single numerical final answer.\n\n"
        f"Problem:\n{question}\n\n"
        "You may use and extend the following reasoning (THOUGHTS):\n"
        f"{thoughts.strip() or '(none)'}\n\n"
        "Briefly continue the reasoning if needed, then compute the final answer.\n"
        "At the END, output exactly one line as the LAST line in this format:\n"
        "FINAL_ANSWER: <value>\n"
        "- <value> should be a single number or a simple numeric expression (like a fraction).\n"
        "- Do NOT output anything after the FINAL_ANSWER line.\n"
    )

def tot_search_for_task(
    tokenizer,
    model,
    question: str,
    cfg: ToTConfig,
) -> Tuple[str, Dict[str, Any]]:
    """
    Run a small Tree-of-Thoughts-style search for a single math question.

    Returns:
        best_rollout_text (may be empty if everything failed),
        debug dict with frontier values, answer_scores, rollout stats, and token/time usage.
    """
    total_gen_seconds = 0.0
    total_out_tokens = 0

    frontier: List[ThoughtState] = [ThoughtState(thoughts="", value=0.0)]

    # Expansion phase
    for depth in range(cfg.max_depth):
        new_candidates: List[ThoughtState] = []
        for state_idx, state in enumerate(frontier):
            for b in range(cfg.branch_factor):
                # 1) Propose next thought
                prop_prompt_raw = _tot_propose_prompt_math(question, state.thoughts)
                prop_prompt = apply_chat_template(tokenizer, prop_prompt_raw)
                prop_out, dt_p, tok_p = generate_one(
                    tokenizer,
                    model,
                    prop_prompt,
                    max_new_tokens=cfg.max_new_tokens_thought,
                    temperature=cfg.temperature_thought,
                    top_p=cfg.top_p_thought,
                    do_sample=True,
                    stop_on_final_answer=False,
                )
                total_gen_seconds += dt_p
                total_out_tokens += tok_p

                lines = [ln for ln in prop_out.splitlines() if ln.strip()]
                if lines and lines[-1].strip().upper() == "CONTINUE":
                    lines = lines[:-1]
                thought_step = "\n".join(lines).strip()
                if not thought_step:
                    continue

                if state.thoughts.strip():
                    new_thoughts = state.thoughts.strip() + "\n" + thought_step
                else:
                    new_thoughts = thought_step

                # 2) Value new state
                val_prompt_raw = _tot_value_prompt_math(question, new_thoughts)
                val_prompt = apply_chat_template(tokenizer, val_prompt_raw)
                val_out, dt_v, tok_v = generate_one(
                    tokenizer,
                    model,
                    val_prompt,
                    max_new_tokens=cfg.max_new_tokens_value,
                    temperature=cfg.temperature_value,
                    top_p=1.0,
                    do_sample=False,
                    stop_on_final_answer=False,
                )
                total_gen_seconds += dt_v
                total_out_tokens += tok_v

                score = _parse_score(val_out)
                if score is None:
                    score = 5  # neutral fallback

                new_candidates.append(ThoughtState(thoughts=new_thoughts, value=float(score)))

        if not new_candidates:
            break

        new_candidates.sort(key=lambda s: s.value, reverse=True)
        frontier = new_candidates[: cfg.beam_size]

    # Rollout phase
    answer_scores: Dict[str, float] = {}
    all_rollouts: List[Dict[str, Any]] = []
    num_rollouts = 0

    for st_idx, st in enumerate(frontier):
        for r in range(cfg.n_rollouts):
            num_rollouts += 1
            roll_prompt_raw = _tot_rollout_prompt_math(question, st.thoughts)
            roll_prompt = apply_chat_template(tokenizer, roll_prompt_raw)
            roll_out, dt_r, tok_r = generate_one(
                tokenizer,
                model,
                roll_prompt,
                max_new_tokens=cfg.max_new_tokens_rollout,
                temperature=cfg.temperature_rollout,
                top_p=cfg.top_p_rollout,
                do_sample=True,
                stop_on_final_answer=True,
            )
            total_gen_seconds += dt_r
            total_out_tokens += tok_r

            ans_tok = _canon(_extract_token(roll_out))
            if ans_tok is not None:
                answer_scores.setdefault(ans_tok, 0.0)
                answer_scores[ans_tok] += float(st.value)

            all_rollouts.append({
                "state_index": st_idx,
                "state_value": st.value,
                "answer_token": ans_tok,
                "rollout_text": roll_out,
            })

    best_rollout_text = ""
    if answer_scores:
        best_token = max(answer_scores.items(), key=lambda kv: kv[1])[0]
        for r in all_rollouts:
            if r.get("answer_token") == best_token and r.get("rollout_text"):
                best_rollout_text = r["rollout_text"]
                break

    if not best_rollout_text and all_rollouts:
        best_rollout_text = all_rollouts[0].get("rollout_text", "")

    debug = {
        "frontier_states": [{"thoughts": st.thoughts, "value": st.value} for st in frontier],
        "answer_scores": answer_scores,
        "num_rollouts": num_rollouts,
        "tot_total_gen_seconds": total_gen_seconds,
        "tot_total_out_tokens": total_out_tokens,
    }
    return best_rollout_text, debug

# ===================== Main =====================

def main():
    if not os.path.exists(BENCH_PATH):
        raise FileNotFoundError(BENCH_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"== Loading: {MODEL_NAME} (device={DEVICE}, dtype={DTYPE})")
    tok, mdl, mode = load_model(MODEL_NAME)

    tasks = read_math_benchmark_jsonl(BENCH_PATH)
    if not tasks:
        raise RuntimeError("No tasks found.")

    mtag = _sanitize_tag(MODEL_NAME.split("/")[-1])
    run_dir = os.path.join(OUT_DIR, f"{mtag}_{mode}")
    os.makedirs(run_dir, exist_ok=True)

    done = {fn[len("task_"):-5] for fn in os.listdir(run_dir)
            if fn.startswith("task_") and fn.endswith(".json")} if RESUME_RUN else set()

    pending = [(tid, q, a) for (tid, q, a) in tasks if not (RESUME_RUN and tid in done)]
    if TASK_LIMIT is not None:
        pending = pending[:TASK_LIMIT]

    start = time.time()
    seen = 0
    ttime = 0.0
    outtok = 0
    gold_num_seen = 0
    num_ok = 0
    partials_path = os.path.join(run_dir, "partials.jsonl")

    print(f"Generating {len(pending)} answers with ToT... report every {REPORT_EVERY}")
    for tid, q, gold_raw in pending:
        # ---- 1. ToT search ----
        tot_ans, tot_debug = tot_search_for_task(tok, mdl, q, TOT_CONFIG)
        tot_seconds = tot_debug.get("tot_total_gen_seconds", 0.0)
        tot_tokens = tot_debug.get("tot_total_out_tokens", 0)
        used_method = "tot"

        ans = tot_ans or ""
        dt = tot_seconds
        out_tok = tot_tokens

        # ---- 2. Fallback to single-shot baseline if ToT produced nothing ----
        if not ans.strip():
            used_method = "baseline_fallback"
            sgen = stream_generate(tok, mdl, [q], MAX_NEW_TOKENS_BASELINE, 1)
            _idx, base_ans, base_dt, base_out_tok = next(sgen)
            ans = base_ans
            dt = tot_seconds + base_dt
            out_tok = tot_tokens + base_out_tok

        # ---- 3. Extract numeric final token & enforce FINAL_ANSWER line ----
        model_tok = _canon(_extract_token(ans))
        if model_tok:
            last = _last_nonempty_line(ans)
            m_ours = FINAL_TAG_RE.search(last)
            ok = bool(m_ours and _canon(m_ours.group(1)) == model_tok)
            ans_marked = ans if ok else (ans.rstrip() + ("\n" if not ans.endswith("\n") else "") + f"{FINAL_TAG} {model_tok}")
        else:
            ans_marked = ans.rstrip() + ("\n" if not ans.endswith("\n") else "") + f"{FINAL_TAG} "

        # ---- 4. Gold token & numeric comparison ----
        gold_tok_raw = (
            hash4_pat.search(gold_raw).group(1).strip()
            if (gold_raw and hash4_pat.search(gold_raw))
            else _extract_token(gold_raw or "")
        )
        gold_tok = _canon(gold_tok_raw)

        def _to_float(tok: Optional[str]) -> Optional[float]:
            if tok is None:
                return None
            t = _norm_num(tok)
            try:
                return float(t)
            except Exception:
                return None

        def _num_from_text(s: Optional[str]) -> Optional[float]:
            if not s:
                return None
            mf = frac_pat.search(s)
            if mf:
                a, b = (mf.group(1), mf.group(2)) if mf.group(1) else (mf.group(3), mf.group(4))
                try:
                    return float(_norm_num(a)) / float(_norm_num(b))
                except Exception:
                    pass
            n = _find_last_number(s)
            try:
                return float(_norm_num(n)) if n else None
            except Exception:
                return None

        gnum = _to_float(gold_tok) or _num_from_text(gold_raw)
        pnum = _to_float(model_tok) or _num_from_text(ans)
        if (gnum is not None) and (pnum is not None):
            gold_num_seen += 1
            if math.isclose(gnum, pnum, rel_tol=1e-6, abs_tol=1e-8):
                num_ok += 1

        # ---- 5. Save per-task JSON ----
        rec = {
            "task_id": tid,
            "question": q,
            "gold_field_raw": gold_raw,
            "model_answer_raw": ans_marked,
            "gold_final_token": gold_tok,
            "model_final_token": model_tok,
            "model_name": MODEL_NAME,
            "loader_mode": mode,
            "search_method": used_method,
            "tot_config": TOT_CONFIG.__dict__,
            "tot_debug": {
                "answer_scores": tot_debug.get("answer_scores", {}),
                "frontier_states": tot_debug.get("frontier_states", []),
                "num_rollouts": tot_debug.get("num_rollouts", 0),
            },
            "max_new_tokens_start": MAX_NEW_TOKENS_BASELINE,
            "batch_size_start": BATCH_SIZE,
            "gen_seconds": dt,
            "out_tokens": out_tok,
        }
        with open(os.path.join(run_dir, f"task_{tid}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        seen += 1
        ttime += dt
        outtok += out_tok

        if seen % REPORT_EVERY == 0:
            el = time.time() - start
            num_acc = (num_ok / gold_num_seen) if gold_num_seen else 0.0
            nm_lo, nm_hi = _wilson_ci(num_ok, gold_num_seen) if gold_num_seen else (0.0, 0.0)
            print(
                f"[{seen}] Num {num_acc*100:.1f}% (n={gold_num_seen}, 95% CI {nm_lo*100:.1f}–{nm_hi*100:.1f}); "
                f"avg out tok {outtok/seen:.1f}; qps {seen/max(el,1e-9):.2f}; mode={mode}"
            )
            with open(partials_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps({
                    "seen": seen,
                    "gold_num_seen": gold_num_seen,
                    "num_correct": num_ok,
                    "num_acc": num_acc,
                    "num_ci95": [nm_lo, nm_hi],
                    "avg_out_tokens": outtok/seen,
                    "throughput_qps": seen/max(el, 1e-9),
                    "elapsed_seconds": el,
                    "model_name": MODEL_NAME,
                    "loader_mode": mode,
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
            "num_correct": num_ok,
            "tot_config": TOT_CONFIG.__dict__,
        }, f, ensure_ascii=False, indent=2)

    del mdl, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Saved per-task outputs in: {run_dir}\nPartials: {partials_path}")

if __name__ == "__main__":
    main()
