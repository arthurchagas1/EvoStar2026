# -*- coding: utf-8 -*-
"""
MMLU ToT baseline — Qwen/Qwen2.5-7B-Instruct (MCQ A–D, FINAL_ANSWER line)

- Same MMLU CSV loader / formatting as your Evo-Reasoner MCQ4 script:
  * read_mmlu_csvs(...) over /data/MMMLU/test
  * build_mcq_task_block(...) builds the question+options block
- Uses robust MCQ letter canonicalization and FINAL_ANSWER extraction logic.
- Evaluates exact-match EM over letters (A/B/C/D).
- Saves one JSON per task + a summary.json + partials.jsonl.

New vs your simple baseline:
- Adds a lightweight Tree-of-Thoughts (ToT) style search per question:
  * Propose: expand partial THOUGHTS into multiple candidate next thoughts.
  * Value: score each partial reasoning path (1–10) for promise.
  * Select: keep a small beam of best states across depths.
  * Rollout: from each top state, finish reasoning and commit to FINAL_ANSWER.
  * Aggregate votes over rollouts, weighted by state values, to pick the final letter.
- Includes a greedy single-shot fallback baseline when ToT fails to produce a letter.

Note: This is a test-time search algorithm; no training is performed.
"""

import os, re, csv, json, math, time, gc
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from tqdm.auto import tqdm

# PyTorch allocator config
os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:512,expandable_segments:true")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

try:
    from transformers.utils import is_flash_attn_2_available
    _HAS_FLASH2 = is_flash_attn_2_available()
except Exception:
    _HAS_FLASH2 = False

# ===================== Config =====================

MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
OUT_DIR    = "tot_deepseek_7b_runs_MMMLU"

# Same folder used in your Evo-Reasoner MCQ script
BENCH_DIR_MMLU = "/scratch/pedro.bento/evostar/data/MMMLU/test"

TASK_LIMIT   = 100       # cap to first 100 tasks (like Evo script)
RESUME_RUN   = True

# Baseline (fallback) decoding budget
BASELINE_MAX_NEW_TOKENS = 8

# ToT search budget (per question)
@dataclass
class ToTConfig:
    max_depth: int = 2            # number of expansion levels
    branch_factor: int = 2        # children per state
    beam_size: int = 3            # states kept per depth
    n_rollouts: int = 2           # rollouts per frontier state

    max_new_tokens_thought: int = 64
    max_new_tokens_value: int = 48
    max_new_tokens_rollout: int = 96

    temperature_thought: float = 0.7
    temperature_value: float = 0.0   # deterministic for scoring
    temperature_rollout: float = 0.7

    top_p_thought: float = 0.9
    top_p_rollout: float = 0.9

TOT_CONFIG = ToTConfig()

BATCH_SIZE     = 16      # only used by the old streaming baseline (kept for convenience)
REPORT_EVERY   = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

# HF cache & offload dirs (speeds up reloads)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", CACHE_DIR)

OFFLOAD_DIR = os.path.join(OUT_DIR, "offload_cache")
os.makedirs(OFFLOAD_DIR, exist_ok=True)

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

ATTN_IMPL = "flash_attention_2" if _HAS_FLASH2 else "sdpa"

FINAL_TAG = "FINAL_ANSWER:"

# ===================== MCQ extraction helpers =====================

MCQ_LETTER_RE      = re.compile(r"^[A-D]$", re.I)
our_final_line_pat = re.compile(r"(?im)^\s*FINAL_ANSWER\s*:\s*(.+?)\s*$")
final_line_pat     = re.compile(r"(?im)^\s*(?:final\s*answer|answer|result)\s*[:=]\s*([A-D]|.+?)\s*$")
rhet_pat           = re.compile(
    r"(?i)(?:thus|therefore|so|hence|consequently)[, ]+(?:the )?answer(?: is)?\s*[:=]?\s*([A-D]|[^\n\r]+)"
)
TAG_RE             = re.compile(r"<(SETUP|PLAN|SOLVE|CHECK|ANSWER|META)>(.*?)</\1>", re.S | re.I)
_mcq_letter_re     = re.compile(r"\b([ABCD])\b", re.I)


def get_sections(text: str) -> Dict[str, str]:
    return {m.group(1).upper(): m.group(2).strip() for m in TAG_RE.finditer(text or "")}


def _canon_mcq(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    t = token.strip().upper()
    return t if MCQ_LETTER_RE.fullmatch(t) else None


def _extract_token_mcq(s: str) -> Optional[str]:
    if not s:
        return None

    # Prefer explicit FINAL_ANSWER line
    m = our_final_line_pat.findall(s)
    if m:
        c = _canon_mcq(m[-1])
        if c:
            return c

    # Generic "answer/result" lines
    m2 = final_line_pat.findall(s)
    if m2:
        c = _canon_mcq(m2[-1])
        if c:
            return c

    # Rhetorical "the answer is X"
    m3 = list(rhet_pat.finditer(s))
    if m3:
        c = _canon_mcq(m3[-1].group(1))
        if c:
            return c

    # Last resort: any standalone A/B/C/D, take the last one
    m4 = _mcq_letter_re.findall(s)
    if m4:
        return m4[-1].upper()

    return None


def _extract_token_token(s: str) -> Optional[str]:
    # For this baseline we only care about MCQ letters
    return _extract_token_mcq(s)


def _canon_token(token: Optional[str]) -> Optional[str]:
    return _canon_mcq(token)


def _last_nonempty_line(text: str) -> str:
    for line in reversed((text or "").splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _answer_line_from_text(text: str) -> str:
    secs = get_sections(text)
    tok = _canon_token(secs.get("ANSWER", "").strip()) or _canon_token(_extract_token_token(text or ""))
    return f"{FINAL_TAG} {tok}" if tok else f"{FINAL_TAG} "


# ===================== Task normalization / formatting =====================

def normalize_task(q: str) -> str:
    q = re.sub(r"\bHow\s+load\b", "How long", q, flags=re.I)
    q = q.replace("mins", "minutes")
    return q


def build_mcq_task_block(question: str, choiceA: str, choiceB: str, choiceC: str, choiceD: str) -> str:
    """Same shape as in Evo-Reasoner: question + Options A–D."""
    q = normalize_task(question)

    def clean(x: str) -> str:
        return (x or "").replace("\r", " ").strip()

    A, B, C, D = map(clean, [choiceA, choiceB, choiceC, choiceD])
    return (
        "This is a MULTIPLE-CHOICE question. Choose exactly one option: A, B, C, or D.\n\n"
        f"Question:\n{q}\n\n"
        "Options:\n"
        f"A) {A}\n"
        f"B) {B}\n"
        f"C) {C}\n"
        f"D) {D}\n"
        "\nAnswer the question and then output ONLY the letter of the correct option."
    )


def read_mmlu_csvs(folder: str, files_limit: int = 10, rows_limit: int = 10) -> List[Tuple[str, str, str]]:
    """
    Returns a list of (task_id, formatted_task_text, gold_letter).

    Same behaviour as in the Evo-Reasoner MCQ4 script:
    - Sort CSV files; take first `files_limit`.
    - From each file, take first `rows_limit` non-header rows.
    - Expect columns: question, A, B, C, D, answer.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"MMLU directory not found: {folder}")

    files = sorted([fn for fn in os.listdir(folder) if fn.lower().endswith(".csv")])[:files_limit]
    tasks: List[Tuple[str, str, str]] = []

    for fn in files:
        path = os.path.join(folder, fn)
        subject = os.path.splitext(fn)[0]
        with open(path, "r", encoding="utf-8", newline="") as fp:
            reader = csv.reader(fp)
            row_idx = 0
            for row in reader:
                if not row:
                    continue
                # Skip header
                if row_idx == 0 and len(row) >= 6 and row[0].strip().lower() in ("question", "pergunta"):
                    row_idx += 1
                    continue
                if len(row) < 6:
                    continue

                q, A, B, C, D, ans = row[0], row[1], row[2], row[3], row[4], row[5]
                ans = (ans or "").strip().upper()
                if not MCQ_LETTER_RE.fullmatch(ans):
                    # some variants might have E or other junk; skip to stay MCQ4
                    continue

                task_text = build_mcq_task_block(q, A, B, C, D)
                tid = f"{subject}-{row_idx:04d}"
                tasks.append((tid, task_text, ans))
                row_idx += 1

                if row_idx > rows_limit:
                    break

    # Cap to 100 if larger (should be exactly 10*10)
    return tasks[:100]


# ===================== Loader (full GPU bf16 first) =====================

def _common_model_kwargs(torch_dtype=None, device_map=None, quantization_config=None):
    kw = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        cache_dir=CACHE_DIR,
    )
    if torch_dtype is not None:
        kw["torch_dtype"] = torch_dtype
    if device_map is not None:
        kw["device_map"] = device_map
    if quantization_config is not None:
        kw["quantization_config"] = quantization_config
    kw["attn_implementation"] = ATTN_IMPL
    return kw


def _try_full_gpu(name, dtype):
    m = AutoModelForCausalLM.from_pretrained(
        name,
        **_common_model_kwargs(torch_dtype=dtype),
    ).to("cuda")
    return m, "full_gpu_bf16"


def _try_auto_offload(name, dtype):
    m = AutoModelForCausalLM.from_pretrained(
        name,
        **_common_model_kwargs(torch_dtype=dtype, device_map="auto"),
        offload_folder=OFFLOAD_DIR,
        offload_state_dict=True,
    )
    return m, "auto_offload_bf16"


def _try_8bit(name):
    if BitsAndBytesConfig is None:
        raise RuntimeError("bitsandbytes unavailable (needed for 8-bit quantization)")
    q = BitsAndBytesConfig(load_in_8bit=True)
    m = AutoModelForCausalLM.from_pretrained(
        name,
        **_common_model_kwargs(quantization_config=q, device_map="auto"),
        offload_folder=OFFLOAD_DIR,
    )
    return m, "8bit_auto"


def _try_4bit(name):
    if BitsAndBytesConfig is None:
        raise RuntimeError("bitsandbytes unavailable (needed for 4-bit quantization)")
    q = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        ),
    )
    m = AutoModelForCausalLM.from_pretrained(
        name,
        **_common_model_kwargs(quantization_config=q, device_map="auto"),
        offload_folder=OFFLOAD_DIR,
    )
    return m, "4bit_auto_nf4"


def load_model(name: str):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch.set_float32_matmul_precision("high")

    last_err: Optional[Exception] = None
    mdl = None
    mode = None

    # 1) Prefer full-GPU bf16 on your 4090 — fastest.
    for loader, label in ((_try_full_gpu, "full_gpu_bf16"), (_try_auto_offload, "auto_offload_bf16")):
        try:
            tqdm.write(f"[Loader] Trying {label} ...")
            mdl, mode = loader(name, DTYPE)
            tqdm.write(f"[Loader]   -> OK ({label})")
            break
        except Exception as e:
            last_err = e
            tqdm.write(f"[Loader]   -> failed ({label}): {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 2) Fallback to 4-bit / 8-bit if needed.
    if mdl is None:
        for loader, label in ((_try_4bit, "4bit_auto_nf4"), (_try_8bit, "8bit_auto")):
            try:
                tqdm.write(f"[Loader] Trying {label} ...")
                mdl, mode = loader(name)
                tqdm.write(f"[Loader]   -> OK ({label})")
                break
            except Exception as e:
                last_err = e
                tqdm.write(f"[Loader]   -> failed ({label}): {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if mdl is None:
        msg = str(last_err) if last_err is not None else ""
        if "torch.load" in msg or "upgrade torch to at least v2.6" in msg:
            raise RuntimeError(
                "Model load failed due to the new torch.load vulnerability guard.\n"
                "This usually means the model repo only ships PyTorch .bin weights.\n"
                "This script forces use_safetensors=True to avoid using torch.load.\n\n"
                "Options:\n"
                "  • Use a safetensors-based checkpoint for this model, or\n"
                "  • Upgrade PyTorch to >= 2.6 if you must load .bin weights.\n"
                f"Last error from HF: {msg}"
            ) from last_err
        raise RuntimeError(f"Model load failed. Last error: {last_err}") from last_err

    if getattr(mdl, "generation_config", None) is not None:
        mdl.generation_config.pad_token_id = tok.eos_token_id
        mdl.generation_config.num_beams = 1

    mdl.eval()
    print(f"[Loader] Using mode: {mode}", flush=True)
    return tok, mdl, mode


def _primary_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ===================== Generic generation helper =====================

def generate_one(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
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
                truncation=True,
            )
            inputs = {k: v.to(dev) for k, v in inputs.items()}

            gen = model.generate(
                **inputs,
                max_new_tokens=cur_max_new,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
            dt = time.time() - t0
            full = tokenizer.decode(gen[0], skip_special_tokens=True)
            if full.startswith(prompt):
                out = full[len(prompt):].strip()
            else:
                out = full.strip()

            out_tok_ids = tokenizer(out, return_tensors=None).input_ids
            out_tok = len(out_tok_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return out, dt, int(out_tok)

        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if cur_max_new > 16:
                cur_max_new = max(16, cur_max_new // 2)
                print(f"[Gen OOM] max_new_tokens -> {cur_max_new}", flush=True)
                continue
            else:
                raise


# ===================== Baseline (fallback) =====================

def build_user_prompt(task_text: str) -> str:
    """
    Short, strict prompt: we let the model continue from 'FINAL_ANSWER: '.
    No chat_template, no explicit chain-of-thought request.
    """
    return (
        f"{task_text}\n\n"
        "Choose the correct option and answer by writing only the letter (A, B, C, or D).\n"
        "FINAL_ANSWER: "
    )


def greedy_baseline_letter(tokenizer, model, task_text: str) -> Tuple[Optional[str], str, float, int]:
    """
    Greedy single-shot baseline, used as a fallback when ToT fails.
    Returns:
        model_token (A-D or None), raw_completion_text, gen_seconds, out_tokens
    """
    prompt = build_user_prompt(task_text)
    out, dt, out_tok = generate_one(
        tokenizer,
        model,
        prompt,
        max_new_tokens=BASELINE_MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
    )
    model_token = _canon_token(_extract_token_token(out))
    # Ensure a FINAL_ANSWER line in raw text
    raw = out.rstrip()
    if not raw.endswith("\n"):
        raw += "\n"
    if model_token:
        raw += f"{FINAL_TAG} {model_token}"
    else:
        raw += f"{FINAL_TAG} "
    return model_token, raw, dt, out_tok


# ===================== ToT components =====================

@dataclass
class ThoughtState:
    thoughts: str
    value: float


def _tot_propose_prompt(task_text: str, thoughts: str) -> str:
    base = (
        "You are solving a multiple-choice question with options A, B, C, and D.\n\n"
        f"{task_text}\n\n"
        "We keep a running reasoning scratchpad called THOUGHTS.\n"
        "THOUGHTS so far:\n"
    )
    body = thoughts.strip() if thoughts.strip() else "(none yet)"
    tail = (
        "\n\nWrite ONE new short step of reasoning that logically continues from THOUGHTS.\n"
        "- Focus only on analysis, do not restate the question.\n"
        "- Do NOT pick a final answer.\n"
        "- End with a line that contains only the word CONTINUE.\n\n"
        "New step:\n"
    )
    return base + body + tail


def _tot_value_prompt(task_text: str, thoughts: str) -> str:
    return (
        "You are evaluating a partial line of reasoning for a multiple-choice question.\n\n"
        f"{task_text}\n\n"
        "Reasoning so far (THOUGHTS):\n"
        f"{thoughts.strip() or '(none)'}\n\n"
        "Based ONLY on how logically sound and promising this reasoning is for finding the correct answer,\n"
        "rate it on a scale from 1 to 10.\n"
        "- 1 = clearly flawed or irrelevant\n"
        "- 10 = highly accurate and almost certainly leads to the correct answer\n\n"
        "Reply with exactly one line in the format:\n"
        "SCORE: <integer from 1 to 10>\n"
    )


def _tot_rollout_prompt(task_text: str, thoughts: str) -> str:
    return (
        "You are solving the following multiple-choice question.\n\n"
        f"{task_text}\n\n"
        "You may use and extend the following reasoning (THOUGHTS):\n"
        f"{thoughts.strip() or '(none)'}\n\n"
        "Briefly continue the reasoning if needed, then decide on the SINGLE best option (A, B, C, or D).\n"
        "At the end, output on its own line:\n"
        "FINAL_ANSWER: <letter>\n"
        "Make sure the FINAL_ANSWER line is present and uses exactly one letter.\n"
    )


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


def tot_search_for_task(
    tokenizer,
    model,
    task_text: str,
    cfg: ToTConfig,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Run a small Tree-of-Thoughts-style BFS for a single MMLU question.

    Returns:
        final_letter (A-D or None),
        debug dict with frontier / rollouts / timing / token stats
    """
    total_gen_seconds = 0.0
    total_out_tokens = 0

    frontier: List[ThoughtState] = [ThoughtState(thoughts="", value=0.0)]
    all_states_debug: List[Dict[str, Any]] = []

    # Expansion phase
    for depth in range(cfg.max_depth):
        new_candidates: List[ThoughtState] = []
        for state_idx, state in enumerate(frontier):
            for b in range(cfg.branch_factor):
                # 1) Propose next thought
                prop_prompt = _tot_propose_prompt(task_text, state.thoughts)
                prop_out, dt_p, tok_p = generate_one(
                    tokenizer,
                    model,
                    prop_prompt,
                    max_new_tokens=cfg.max_new_tokens_thought,
                    temperature=cfg.temperature_thought,
                    top_p=cfg.top_p_thought,
                    do_sample=True,
                )
                total_gen_seconds += dt_p
                total_out_tokens += tok_p

                # Strip trailing CONTINUE line if present
                lines = prop_out.splitlines()
                lines = [ln for ln in lines if ln.strip()]
                if lines and lines[-1].strip().upper() == "CONTINUE":
                    lines = lines[:-1]
                thought_step = "\n".join(lines).strip()
                if not thought_step:
                    # skip degenerate child
                    continue

                # 2) Build new THOUGHTS
                if state.thoughts.strip():
                    new_thoughts = state.thoughts.strip() + "\n" + thought_step
                else:
                    new_thoughts = thought_step

                # 3) Value the new state
                val_prompt = _tot_value_prompt(task_text, new_thoughts)
                val_out, dt_v, tok_v = generate_one(
                    tokenizer,
                    model,
                    val_prompt,
                    max_new_tokens=cfg.max_new_tokens_value,
                    temperature=cfg.temperature_value,
                    top_p=1.0,
                    do_sample=False,
                )
                total_gen_seconds += dt_v
                total_out_tokens += tok_v

                score = _parse_score(val_out)
                if score is None:
                    score = 5  # neutral fallback

                new_state = ThoughtState(thoughts=new_thoughts, value=float(score))
                new_candidates.append(new_state)
                all_states_debug.append({
                    "depth": depth,
                    "parent_index": state_idx,
                    "branch_id": b,
                    "thought_step": thought_step,
                    "thoughts_full": new_thoughts,
                    "score": score,
                    "value_raw_text": val_out,
                })

        if not new_candidates:
            break

        # Beam select best states by value
        new_candidates.sort(key=lambda s: s.value, reverse=True)
        frontier = new_candidates[: cfg.beam_size]

    # Rollout phase
    rollouts_debug: List[Dict[str, Any]] = []
    letter_scores: Dict[str, float] = {}

    for st_idx, st in enumerate(frontier):
        for r in range(cfg.n_rollouts):
            roll_prompt = _tot_rollout_prompt(task_text, st.thoughts)
            roll_out, dt_r, tok_r = generate_one(
                tokenizer,
                model,
                roll_prompt,
                max_new_tokens=cfg.max_new_tokens_rollout,
                temperature=cfg.temperature_rollout,
                top_p=cfg.top_p_rollout,
                do_sample=True,
            )
            total_gen_seconds += dt_r
            total_out_tokens += tok_r

            token = _canon_token(_extract_token_token(roll_out))
            rollouts_debug.append({
                "state_index": st_idx,
                "state_value": st.value,
                "state_thoughts": st.thoughts,
                "rollout_index": r,
                "rollout_text": roll_out,
                "answer_token": token,
            })
            if token is not None:
                letter_scores.setdefault(token, 0.0)
                # Weight by state value; could also add +1 vote baseline
                letter_scores[token] += float(st.value)

    final_letter: Optional[str] = None
    if letter_scores:
        final_letter = max(letter_scores.items(), key=lambda kv: kv[1])[0]

    debug = {
        "frontier_states": [{"thoughts": st.thoughts, "value": st.value} for st in frontier],
        "all_states": all_states_debug,
        "rollouts": rollouts_debug,
        "letter_scores": letter_scores,
        "tot_total_gen_seconds": total_gen_seconds,
        "tot_total_out_tokens": total_out_tokens,
    }
    return final_letter, debug


# ===================== Stats helpers =====================

def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


# ===================== Main =====================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"== Loading: {MODEL_NAME} (device={DEVICE}, dtype={DTYPE})", flush=True)
    tok, mdl, mode = load_model(MODEL_NAME)

    bench = read_mmlu_csvs(BENCH_DIR_MMLU, files_limit=10, rows_limit=10)
    if not bench:
        raise RuntimeError("No tasks found in MMLU folder.")

    run_dir = os.path.join(
        OUT_DIR,
        f"mmlu_{MODEL_NAME.split('/')[-1]}_{mode}"
    )
    os.makedirs(run_dir, exist_ok=True)

    # RESUME: skip tasks that already have task_<id>.json
    done_ids = set()
    if RESUME_RUN:
        for fn in os.listdir(run_dir):
            if fn.startswith("task_") and fn.endswith(".json"):
                done_ids.add(fn[len("task_"):-5])

    pending = [(tid, task_text, gold) for (tid, task_text, gold) in bench if tid not in done_ids]
    if TASK_LIMIT is not None:
        pending = pending[:TASK_LIMIT]

    print(f"Total tasks loaded: {len(bench)} | Pending: {len(pending)}", flush=True)

    total_seen = 0
    total_time = 0.0
    total_out_tokens = 0
    gold_seen = 0
    num_correct = 0
    start = time.time()
    partials_path = os.path.join(run_dir, "partials.jsonl")

    for idx, (tid, task_text, gold_letter) in enumerate(pending):
        # Run ToT search
        final_letter, tot_debug = tot_search_for_task(tok, mdl, task_text, TOT_CONFIG)
        tot_seconds = tot_debug["tot_total_gen_seconds"]
        tot_tokens = tot_debug["tot_total_out_tokens"]

        used_method = "tot"
        model_token: Optional[str] = final_letter
        model_answer_raw: str

        # If ToT fails to produce a letter, fall back to greedy baseline
        if model_token is None:
            used_method = "baseline_fallback"
            base_token, base_raw, base_dt, base_tok = greedy_baseline_letter(tok, mdl, task_text)
            model_token = base_token
            model_answer_raw = base_raw
            gen_seconds = tot_seconds + base_dt
            out_tokens = int(tot_tokens + base_tok)
        else:
            # Use the earliest rollout that matches the chosen letter, or just the first rollout
            chosen_rollout_text = None
            for r in tot_debug["rollouts"]:
                if r.get("answer_token") == model_token:
                    chosen_rollout_text = r.get("rollout_text")
                    break
            if chosen_rollout_text is None and tot_debug["rollouts"]:
                chosen_rollout_text = tot_debug["rollouts"][0].get("rollout_text", "")
            elif chosen_rollout_text is None:
                chosen_rollout_text = ""

            # Ensure FINAL_ANSWER line present and consistent
            raw = chosen_rollout_text.rstrip()
            if not raw.endswith("\n"):
                raw += "\n"
            raw += f"{FINAL_TAG} {model_token}"
            model_answer_raw = raw
            gen_seconds = tot_seconds
            out_tokens = int(tot_tokens)

        gold_final_token = _canon_mcq(gold_letter)

        rec = {
            "task_id": tid,
            "question": task_text,
            "gold_field_raw": gold_letter,
            "gold_final_token": gold_final_token,
            "model_answer_raw": model_answer_raw,
            "model_final_token": model_token,
            "model_name": MODEL_NAME,
            "loader_mode": mode,
            "search_method": used_method,
            "tot_config": TOT_CONFIG.__dict__,
            "tot_debug": {
                "letter_scores": tot_debug.get("letter_scores", {}),
                "frontier_states": tot_debug.get("frontier_states", []),
            },
            "max_new_tokens_baseline": BASELINE_MAX_NEW_TOKENS,
            "gen_seconds": gen_seconds,
            "out_tokens": out_tokens,
        }

        with open(os.path.join(run_dir, f"task_{tid}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        total_seen += 1
        total_time += gen_seconds
        total_out_tokens += out_tokens

        if gold_final_token is not None:
            gold_seen += 1
            if model_token is not None and gold_final_token == model_token:
                num_correct += 1

        if total_seen % REPORT_EVERY == 0:
            elapsed = time.time() - start
            acc = (num_correct / gold_seen) if gold_seen else 0.0
            lo, hi = _wilson_ci(num_correct, gold_seen) if gold_seen else (0.0, 0.0)
            avg_tok = total_out_tokens / total_seen if total_seen else 0.0
            tput = total_seen / max(elapsed, 1e-9)
            print(
                f"[{total_seen}] EM {acc*100:.1f}% "
                f"(n={gold_seen}, 95% CI {lo*100:.1f}–{hi*100:.1f}); "
                f"avg out tok {avg_tok:.1f}; throughput {tput:.2f}/s; mode={mode}",
                flush=True,
            )
            with open(partials_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps({
                    "seen": total_seen,
                    "gold_seen": gold_seen,
                    "num_correct": num_correct,
                    "num_acc": acc,
                    "num_ci95": [lo, hi],
                    "avg_out_tokens": avg_tok,
                    "throughput_qps": tput,
                    "model_name": MODEL_NAME,
                    "loader_mode": mode,
                }, ensure_ascii=False) + "\n")

        print(
            f"[Task {tid}] method={used_method} | FINAL_ANSWER={model_token} "
            f"| gold={gold_final_token} | gen_seconds={gen_seconds:.2f}",
            flush=True,
        )

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": "MMLU:test",
            "model_name": MODEL_NAME,
            "device": DEVICE,
            "dtype": str(DTYPE),
            "benchmark_path": BENCH_DIR_MMLU,
            "total": total_seen,
            "timing_seconds": total_time,
            "loader_mode": mode,
            "gold_num_seen": gold_seen,
            "num_correct": num_correct,
            "num_acc": (num_correct / gold_seen) if gold_seen else None,
            "tot_config": TOT_CONFIG.__dict__,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSaved per-task outputs in: {run_dir}\nPartials: {partials_path}", flush=True)


if __name__ == "__main__":
    main()
