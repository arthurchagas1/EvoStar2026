# -*- coding: utf-8 -*-
"""
Usage: python baseline_qwen_got_mmlu.py

Focus:
- Exact-match accuracy on MMLU (MCQ A–D) + reasonable throughput.
- Robust loader: download full snapshot to a local folder, load locally (OOM-safe).
- Graph-of-Thought (GoT) style search for better reasoning:
  * Nodes represent partial THOUGHTS about the MMLU question.
  * Graph expands via local proposals; nodes can persist across depths.
  * Aggregation nodes merge reasoning from multiple parents.
  * Rollout: from high-value nodes, complete reasoning and produce FINAL_ANSWER: <letter>.
- If GoT fails, fall back to a simple, single-shot MCQ baseline.
"""

import os, re, csv, json, math, time, gc
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

MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
BENCH_DIR_MMLU = "/scratch/pedro.bento/evostar/data/MMMLU/test"
OUT_DIR = "/scratch/pedro.bento/evostar/runs/mmlu/got_deepseek_7b_runs_MMMLU"

MAX_NEW_TOKENS_BASELINE = 32   # baseline fallback (few tokens needed for letter)
BATCH_SIZE = 8
REPORT_EVERY = 10
RESUME_RUN = True
TASK_LIMIT = 100  # cap to first 100 tasks (like your Evo script)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Safer DTYPE selection across PyTorch versions
if DEVICE == "cuda":
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        DTYPE = torch.bfloat16
    else:
        # fall back to fp16 if bfloat16 is not supported
        DTYPE = torch.float16
else:
    DTYPE = torch.float32

OFFLOAD_DIR = os.path.join(OUT_DIR, "offload_cache")
LOCAL_MODELS = os.path.join(OUT_DIR, "local_models")
os.makedirs(OFFLOAD_DIR, exist_ok=True)
os.makedirs(LOCAL_MODELS, exist_ok=True)

FINAL_TAG = "FINAL_ANSWER:"
FINAL_TAG_RE = re.compile(r"(?im)^\s*FINAL_ANSWER\s*:\s*(.+?)\s*$")

# ===================== GoT configuration (MCQ, MMLU) =====================

@dataclass
class GoTConfig:
    max_depth: int = 2            # expansion levels
    branch_factor: int = 2        # children per node
    beam_size: int = 3            # nodes kept per depth
    n_rollouts: int = 2           # rollouts per final node

    max_new_tokens_thought: int = 96
    max_new_tokens_value: int = 64
    max_new_tokens_rollout: int = 192

    temperature_thought: float = 0.7
    temperature_value: float = 0.0   # deterministic scoring
    temperature_rollout: float = 0.7

    top_p_thought: float = 0.9
    top_p_rollout: float = 0.9

    # Graph-specific knobs
    max_nodes_for_aggregation: int = 3   # how many top nodes to merge per depth (0 = disable)
    parent_influence: float = 0.3        # weight of parent value in ranking

GOT_CONFIG = GoTConfig()

# ===================== MMLU IO: loader & formatting =====================

def normalize_task(q: str) -> str:
    # Small normalizations you used in your baseline (keep behaviour aligned)
    q = re.sub(r"\bHow\s+load\b", "How long", q, flags=re.I)
    q = q.replace("mins", "minutes")
    return q


def build_mcq_task_block(question: str, choiceA: str, choiceB: str, choiceC: str, choiceD: str) -> str:
    """Same shape as in your Evo-Reasoner MCQ4 and simple baseline: question + Options A–D."""
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

    Behaviour matches your simple baseline:
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
                if not re.fullmatch(r"[ABCD]", ans):
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

# ===================== Loader (snapshot + OOM fallbacks) =====================

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
        local_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=dtype,
        local_files_only=True,
    ).to("cuda")
    return m, "full_gpu_bf16" if dtype == torch.bfloat16 else "full_gpu_fp16" if dtype == torch.float16 else "full_gpu_fp32"


def _mdl_auto_offload(local_path: str, dtype):
    m = AutoModelForCausalLM.from_pretrained(
        local_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=dtype,
        device_map="auto",
        offload_folder=OFFLOAD_DIR,
        offload_state_dict=True,
        local_files_only=True,
    )
    return m, "auto_offload_bf16" if dtype == torch.bfloat16 else "auto_offload_fp16" if dtype == torch.float16 else "auto_offload_fp32"


def _mdl_8bit(local_path: str):
    if BitsAndBytesConfig is None:
        raise RuntimeError("bitsandbytes unavailable")
    q = BitsAndBytesConfig(load_in_8bit=True)
    m = AutoModelForCausalLM.from_pretrained(
        local_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=q,
        device_map="auto",
        offload_folder=OFFLOAD_DIR,
        local_files_only=True,
    )
    return m, "8bit_auto"


def _mdl_4bit(local_path: str):
    if BitsAndBytesConfig is None:
        raise RuntimeError("bitsandbytes unavailable")
    q = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if (torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()) else torch.float16
        ),
    )
    m = AutoModelForCausalLM.from_pretrained(
        local_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=q,
        device_map="auto",
        offload_folder=OFFLOAD_DIR,
        local_files_only=True,
    )
    return m, "4bit_auto_nf4"


def _missing_file_err(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        ("no such file or directory" in msg)
        or ("does not appear to have files named" in msg)
        or ("no file named model.safetensors" in msg)
    )


def load_model(name: str):
    # 1) Download full snapshot locally
    local_path = _snapshot(name, force=False)

    # 2) Tokenizer
    tok = _tok_from_local(local_path)

    # 3) Try load paths (OOM-safe). If missing shards, force re-download once and retry.
    torch.set_float32_matmul_precision("high")

    def _try_all(local_path_inner: str):
        last_err = None
        # High-precision first
        for ctor in (_mdl_full_gpu, _mdl_auto_offload):
            try:
                mdl, mode = ctor(local_path_inner, DTYPE)
                return mdl, mode
            except Exception as e:
                last_err = e
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # Then quantized
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

def apply_chat_template(tokenizer, user_text: str) -> str:
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                add_generation_prompt=True,
                tokenize=False
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

# ===================== Context-length helpers =====================

def _get_ctx_limit(tokenizer, model) -> int:
    """
    Best-effort context window detection with sane fallbacks.
    """
    cfg = getattr(model, "config", None)
    ctx = None
    if cfg is not None and hasattr(cfg, "max_position_embeddings"):
        ctx = int(getattr(cfg, "max_position_embeddings"))
    if ctx is None or ctx <= 0 or ctx > 10_000_000:
        ctx = getattr(tokenizer, "model_max_length", 2048)
    if ctx is None or ctx <= 0 or ctx > 10_000_000:
        ctx = 4096
    return int(ctx)


def _truncate_for_context(
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
    tokenizer,
    model,
    margin: int = 64,
) -> Dict[str, torch.Tensor]:
    """
    Truncate input_ids/attention_mask from the left so that
    prompt_len + max_new_tokens + margin <= context_limit.
    """
    ctx_limit = _get_ctx_limit(tokenizer, model)
    allowed = ctx_limit - max_new_tokens - margin
    if allowed <= 0:
        # still try to keep some space for new tokens
        allowed = max(1, ctx_limit - max_new_tokens)
    if "input_ids" not in inputs:
        return inputs

    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    if seq_len > allowed:
        start = seq_len - allowed
        inputs["input_ids"] = input_ids[:, start:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, start:]
    return inputs

# ===================== Generic generation helper (for GoT & baseline) =====================

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
    Single-prompt generation, with simple OOM backoff and context-safe truncation.

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
            inputs = _truncate_for_context(inputs, cur_max_new, tokenizer, model)
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

# ===================== Baseline MCQ generation (fallback) =====================

def build_user_prompt_baseline(task_text: str) -> str:
    """
    Short, strict MCQ prompt for fallback baseline:
    we let the model continue from 'FINAL_ANSWER: '.
    """
    return (
        f"{task_text}\n\n"
        "Choose the correct option and answer by writing only the letter (A, B, C, or D).\n"
        "Write your final answer in EXACTLY one line in the format:\n"
        "FINAL_ANSWER: <letter>\n"
        "where <letter> is one of A, B, C, or D.\n"
        "FINAL_ANSWER: "
    )


def stream_generate(tokenizer, model, tasks: List[str],
                    max_new_tokens: int, batch_size: int):
    """
    Yield: (index, output_text, gen_seconds, out_token_count)
    tasks: list of task_text blocks (as returned by build_mcq_task_block).
    """
    dev = _primary_device(model)
    micro_bs = max(1, min(batch_size, len(tasks)))
    max_new = max_new_tokens
    i = 0

    with torch.inference_mode():
        while i < len(tasks):
            batch_tasks = tasks[i:i + micro_bs]
            prompts = [build_user_prompt_baseline(t) for t in batch_tasks]
            try:
                t0 = time.time()
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                inputs = {k: v.to(dev) for k, v in inputs.items()}
                inputs = _truncate_for_context(inputs, max_new, tokenizer, model)

                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
                dt = time.time() - t0
                dec = tokenizer.batch_decode(gen, skip_special_tokens=True)

                for j, full in enumerate(dec):
                    pref = prompts[j]
                    out = full[len(pref):].strip() if full.startswith(pref) else full.strip()
                    out_tok_ids = tokenizer(out, return_tensors=None).input_ids
                    out_tok = len(out_tok_ids)
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
                elif max_new > 4:
                    max_new = max(4, max_new // 2)
                    print(f"[Gen OOM] max_new_tokens -> {max_new}")
                    continue
                else:
                    raise

# ===================== MCQ extraction helpers (letters A–D) =====================

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


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))

# ===================== GoT (Graph-of-Thoughts) for MCQ =====================

@dataclass
class GraphNode:
    node_id: int
    thoughts: str
    value: float
    depth: int
    parents: List[int]


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


def _got_propose_prompt_mcq(question_block: str, thoughts: str) -> str:
    base = (
        "You are solving a multiple-choice question with exactly one correct option "
        "(A, B, C, or D).\n\n"
        f"Question and options:\n{question_block}\n\n"
        "We maintain a running reasoning scratchpad called THOUGHTS.\n"
        "THOUGHTS so far:\n"
    )
    body = thoughts.strip() if thoughts.strip() else "(none yet)"
    tail = (
        "\n\nWrite ONE short new step of reasoning that logically continues from THOUGHTS.\n"
        "- Focus only on analysis of the options; do not restate the whole problem.\n"
        "- Do NOT choose the final option yet.\n"
        "- End with a line that contains only the word CONTINUE.\n\n"
        "New step:\n"
    )
    return base + body + tail


def _got_value_prompt_mcq(question_block: str, thoughts: str) -> str:
    return (
        "You are evaluating a partial line of reasoning for a multiple-choice question "
        "(A, B, C, or D).\n\n"
        f"Question and options:\n{question_block}\n\n"
        "Reasoning so far (THOUGHTS):\n"
        f"{thoughts.strip() or '(none)'}\n\n"
        "Based ONLY on how logically correct, relevant, and helpful this reasoning is for "
        "identifying the correct option, rate it on a scale from 1 to 10.\n"
        "- 1 = clearly flawed or irrelevant\n"
        "- 10 = highly accurate and very likely to lead to the correct option\n\n"
        "Reply with exactly one line in the format:\n"
        "SCORE: <integer from 1 to 10>\n"
    )


def _got_rollout_prompt_mcq(question_block: str, thoughts: str) -> str:
    return (
        "You are solving the following multiple-choice question. There is exactly one correct option.\n\n"
        f"Question and options:\n{question_block}\n\n"
        "You may use and extend the following reasoning (THOUGHTS):\n"
        f"{thoughts.strip() or '(none)'}\n\n"
        "Briefly continue the reasoning if needed, then choose the single best option.\n"
        "At the END, output exactly one line as the LAST line in this format:\n"
        "FINAL_ANSWER: <letter>\n"
        "- <letter> must be one of A, B, C, or D.\n"
        "- Do NOT output anything after the FINAL_ANSWER line.\n"
    )


def _got_aggregate_prompt_mcq(question_block: str, nodes_to_merge: List[GraphNode]) -> str:
    parts = []
    for idx, node in enumerate(nodes_to_merge, 1):
        parts.append(f"Reasoning {idx}:\n{(node.thoughts or '').strip() or '(empty)'}")
    body = "\n\n".join(parts)
    return (
        "You are combining multiple partial lines of reasoning for the SAME multiple-choice question.\n\n"
        f"Question and options:\n{question_block}\n\n"
        f"{body}\n\n"
        "Combine the correct and useful elements of these THOUGHTS into a single, improved reasoning trace.\n"
        "- Remove contradictions or obvious mistakes.\n"
        "- Keep the reasoning concise and well ordered.\n"
        "- Do NOT choose the final option yet.\n"
        "End with a line that contains only the word CONTINUE.\n\n"
        "Merged THOUGHTS:\n"
    )


def got_search_for_task(
    tokenizer,
    model,
    question_block: str,
    cfg: GoTConfig,
) -> Tuple[str, Dict[str, Any]]:
    """
    Run a small Graph-of-Thoughts-style search for a single MMLU MCQ question.

    Returns:
        best_rollout_text (may be empty if everything failed),
        debug dict with node values, answer_scores, rollout stats, and token/time usage.
    """
    total_gen_seconds = 0.0
    total_out_tokens = 0

    nodes: Dict[int, GraphNode] = {}
    next_node_id = 0

    # Root node (no thoughts yet)
    root = GraphNode(node_id=next_node_id, thoughts="", value=0.0, depth=0, parents=[])
    nodes[next_node_id] = root
    next_node_id += 1

    frontier: List[int] = [root.node_id]

    # Expansion + aggregation phase
    for depth in range(cfg.max_depth):
        new_ids: List[int] = []

        # 1) Local expansions
        for nid in list(frontier):
            node = nodes[nid]
            for _ in range(cfg.branch_factor):
                prop_prompt_raw = _got_propose_prompt_mcq(question_block, node.thoughts)
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

                if node.thoughts.strip():
                    new_thoughts = node.thoughts.strip() + "\n" + thought_step
                else:
                    new_thoughts = thought_step

                # Score new node
                val_prompt_raw = _got_value_prompt_mcq(question_block, new_thoughts)
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
                    score = 5

                nid_new = next_node_id
                next_node_id += 1
                nodes[nid_new] = GraphNode(
                    node_id=nid_new,
                    thoughts=new_thoughts,
                    value=float(score),
                    depth=node.depth + 1,
                    parents=[nid],
                )
                new_ids.append(nid_new)

        # 2) Aggregation step: merge top-valued nodes (optional)
        if cfg.max_nodes_for_aggregation > 0 and len(nodes) > 1:
            sortable_nodes = [n for n in nodes.values() if n.thoughts.strip()]
            sortable_nodes.sort(key=lambda n: n.value, reverse=True)
            merge_set = sortable_nodes[: cfg.max_nodes_for_aggregation]

            if len(merge_set) >= 2:
                agg_prompt_raw = _got_aggregate_prompt_mcq(question_block, merge_set)
                agg_prompt = apply_chat_template(tokenizer, agg_prompt_raw)
                agg_out, dt_a, tok_a = generate_one(
                    tokenizer,
                    model,
                    agg_prompt,
                    max_new_tokens=cfg.max_new_tokens_thought,
                    temperature=cfg.temperature_thought,
                    top_p=cfg.top_p_thought,
                    do_sample=True,
                    stop_on_final_answer=False,
                )
                total_gen_seconds += dt_a
                total_out_tokens += tok_a

                lines = [ln for ln in agg_out.splitlines() if ln.strip()]
                if lines and lines[-1].strip().upper() == "CONTINUE":
                    lines = lines[:-1]
                agg_thoughts = "\n".join(lines).strip()

                if agg_thoughts:
                    val_prompt_raw = _got_value_prompt_mcq(question_block, agg_thoughts)
                    val_prompt = apply_chat_template(tokenizer, val_prompt_raw)
                    val_out, dt_v2, tok_v2 = generate_one(
                        tokenizer,
                        model,
                        val_prompt,
                        max_new_tokens=cfg.max_new_tokens_value,
                        temperature=cfg.temperature_value,
                        top_p=1.0,
                        do_sample=False,
                        stop_on_final_answer=False,
                    )
                    total_gen_seconds += dt_v2
                    total_out_tokens += tok_v2

                    score2 = _parse_score(val_out)
                    if score2 is None:
                        score2 = 6  # slightly optimistic for merged traces

                    nid_agg = next_node_id
                    next_node_id += 1
                    max_depth_parents = max(n.depth for n in merge_set)
                    nodes[nid_agg] = GraphNode(
                        node_id=nid_agg,
                        thoughts=agg_thoughts,
                        value=float(score2),
                        depth=max_depth_parents + 1,
                        parents=[n.node_id for n in merge_set],
                    )
                    new_ids.append(nid_agg)

        if not new_ids:
            break

        # 3) Choose next frontier using graph-aware ranking
        candidate_ids = list(set(new_ids + frontier))

        def _rank_score(nid: int) -> float:
            node = nodes[nid]
            if node.parents:
                parent_max = max(nodes[p].value for p in node.parents)
            else:
                parent_max = 0.0
            return node.value + cfg.parent_influence * parent_max

        candidate_ids.sort(key=_rank_score, reverse=True)
        frontier = candidate_ids[: cfg.beam_size]

    # Rollout phase
    answer_scores: Dict[str, float] = {}
    all_rollouts: List[Dict[str, Any]] = []
    num_rollouts = 0

    # pick rollout candidates: frontier if they have thoughts, else best nodes overall
    rollout_ids = [nid for nid in frontier if nodes[nid].thoughts.strip()]
    if not rollout_ids:
        rollout_ids = [
            n.node_id for n in sorted(nodes.values(), key=lambda n: n.value, reverse=True)
            if n.thoughts.strip()
        ][: cfg.beam_size]

    for nid in rollout_ids:
        st = nodes[nid]
        for _ in range(cfg.n_rollouts):
            num_rollouts += 1
            roll_prompt_raw = _got_rollout_prompt_mcq(question_block, st.thoughts)
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

            ans_tok = _canon_token(_extract_token_token(roll_out))
            if ans_tok is not None:
                answer_scores.setdefault(ans_tok, 0.0)
                # weight by node value
                answer_scores[ans_tok] += float(st.value)

            all_rollouts.append({
                "node_id": nid,
                "node_value": st.value,
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
        "nodes": [
            {
                "node_id": n.node_id,
                "value": n.value,
                "depth": n.depth,
                "parents": n.parents,
                "has_thoughts": bool(n.thoughts.strip()),
            }
            for n in nodes.values()
        ],
        "frontier_ids": frontier,
        "answer_scores": answer_scores,
        "num_rollouts": num_rollouts,
        "got_total_gen_seconds": total_gen_seconds,
        "got_total_out_tokens": total_out_tokens,
    }
    return best_rollout_text, debug

# ===================== Main =====================

def main():
    if not os.path.exists(BENCH_DIR_MMLU):
        raise FileNotFoundError(BENCH_DIR_MMLU)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"== Loading: {MODEL_NAME} (device={DEVICE}, dtype={DTYPE})")
    tok, mdl, mode = load_model(MODEL_NAME)

    bench = read_mmlu_csvs(BENCH_DIR_MMLU, files_limit=10, rows_limit=10)
    if not bench:
        raise RuntimeError("No tasks found in MMLU folder.")

    run_dir = os.path.join(
        OUT_DIR,
        f"mmlu_got_{MODEL_NAME.split('/')[-1]}_{mode}"
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

    print(f"Total tasks loaded: {len(bench)} | Pending: {len(pending)}")

    start = time.time()
    seen = 0
    ttime = 0.0
    outtok = 0
    gold_num_seen = 0
    num_ok = 0
    partials_path = os.path.join(run_dir, "partials.jsonl")

    print(f"Generating {len(pending)} answers with GoT... report every {REPORT_EVERY}")

    for tid, task_text, gold_letter in pending:
        # ---- 1. GoT search ----
        got_ans, got_debug = got_search_for_task(tok, mdl, task_text, GOT_CONFIG)
        got_seconds = got_debug.get("got_total_gen_seconds", 0.0)
        got_tokens = got_debug.get("got_total_out_tokens", 0)
        used_method = "got"

        ans = got_ans or ""
        dt = got_seconds
        out_tok = got_tokens

        # Try extract token from GoT output
        model_tok = _canon_token(_extract_token_token(ans))

        # ---- 2. Fallback to single-shot baseline if GoT produced nothing
        #          OR if GoT did not yield a valid MCQ letter ----
        if (not ans.strip()) or (model_tok is None):
            used_method = "baseline_fallback" if not ans.strip() else "got_plus_baseline_fallback"
            sgen = stream_generate(tok, mdl, [task_text], MAX_NEW_TOKENS_BASELINE, 1)
            _idx, base_ans, base_dt, base_out_tok = next(sgen)
            ans = base_ans
            dt = got_seconds + base_dt
            out_tok = got_tokens + base_out_tok
            model_tok = _canon_token(_extract_token_token(ans))

        # ---- 3. Enforce FINAL_ANSWER line (for logging) ----
        if model_tok:
            last = _last_nonempty_line(ans)
            m_ours = FINAL_TAG_RE.search(last)
            ok = bool(m_ours and _canon_token(m_ours.group(1)) == model_tok)
            ans_marked = ans if ok else (ans.rstrip() + ("\n" if not ans.endswith("\n") else "") + f"{FINAL_TAG} {model_tok}")
        else:
            ans_marked = ans.rstrip() + ("\n" if not ans.endswith("\n") else "") + f"{FINAL_TAG} "

        out_tokens = int(out_tok)

        # ---- 4. Gold token & accuracy ----
        gold_final_token = _canon_mcq(gold_letter)

        if gold_final_token is not None:
            gold_num_seen += 1
            if model_tok is not None and gold_final_token == model_tok:
                num_ok += 1

        # ---- 5. Save per-task JSON ----
        rec = {
            "task_id": tid,
            "question": task_text,
            "gold_field_raw": gold_letter,
            "gold_final_token": gold_final_token,
            "model_answer_raw": ans_marked,
            "model_final_token": model_tok,
            "model_name": MODEL_NAME,
            "loader_mode": mode,
            "search_method": used_method,
            "got_config": GOT_CONFIG.__dict__,
            "got_debug": {
                "answer_scores": got_debug.get("answer_scores", {}),
                "frontier_ids": got_debug.get("frontier_ids", []),
                "num_rollouts": got_debug.get("num_rollouts", 0),
            },
            "max_new_tokens_start": MAX_NEW_TOKENS_BASELINE,
            "batch_size_start": BATCH_SIZE,
            "gen_seconds": dt,
            "out_tokens": out_tokens,
        }
        with open(os.path.join(run_dir, f"task_{tid}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        seen += 1
        ttime += dt
        outtok += out_tokens

        # ---- 6. Logging & partials ----
        if seen % REPORT_EVERY == 0:
            elapsed = time.time() - start
            num_acc = (num_ok / gold_num_seen) if gold_num_seen else 0.0
            nm_lo, nm_hi = _wilson_ci(num_ok, gold_num_seen) if gold_num_seen else (0.0, 0.0)
            avg_tok = outtok / seen if seen else 0.0
            tput = seen / max(elapsed, 1e-9)
            print(
                f"[{seen}] EM {num_acc*100:.1f}% "
                f"(n={gold_num_seen}, 95% CI {nm_lo*100:.1f}–{nm_hi*100:.1f}); "
                f"avg out tok {avg_tok:.1f}; throughput {tput:.2f}/s; mode={mode}"
            )
            with open(partials_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps({
                    "seen": seen,
                    "gold_num_seen": gold_num_seen,
                    "num_correct": num_ok,
                    "num_acc": num_acc,
                    "num_ci95": [nm_lo, nm_hi],
                    "avg_out_tokens": avg_tok,
                    "throughput_qps": tput,
                    "elapsed_seconds": elapsed,
                    "model_name": MODEL_NAME,
                    "loader_mode": mode,
                }, ensure_ascii=False) + "\n")

        print(
            f"[Task {tid}] FINAL_ANSWER={model_tok} | gold={gold_final_token} | gen_seconds={dt:.2f}",
            flush=True,
        )

    num_acc_final = (num_ok / gold_num_seen) if gold_num_seen else None

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": "MMLU:test",
            "model_name": MODEL_NAME,
            "device": DEVICE,
            "dtype": str(DTYPE),
            "benchmark_path": BENCH_DIR_MMLU,
            "total": seen,
            "timing_seconds": ttime,
            "loader_mode": mode,
            "gold_num_seen": gold_num_seen,
            "num_correct": num_ok,
            "num_acc": num_acc_final,
            "got_config": GOT_CONFIG.__dict__,
        }, f, ensure_ascii=False, indent=2)

    del mdl, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nSaved per-task outputs in: {run_dir}\nPartials: {partials_path}")


if __name__ == "__main__":
    main()
