# -*- coding: utf-8 -*-
"""
MMLU simple baseline — openai/gpt-oss-20b (MCQ A–D, FINAL_ANSWER line)

- Uses the SAME MMLU CSV loader and task formatting as your Evo-Reasoner MCQ4 script:
  * read_mmlu_csvs(...) over /data/MMMLU/test
  * build_mcq_task_block(...) with the same question+options block
- Uses robust MCQ letter canonicalization and FINAL_ANSWER extraction logic.
- Evaluates exact-match EM over letters (A/B/C/D).
- Saves one JSON per task + a summary.json + partials.jsonl.

Key design choices (aligned with your working math baseline):
- Prompt: explains the MCQ, then asks the model to output a line:
    FINAL_ANSWER: <letter>
  where <letter> ∈ {A,B,C,D}, with *no extra text after that line*.
- We apply a FinalAnswerStopper that stops generation as soon as a non-empty
  "FINAL_ANSWER: <...>" line appears anywhere in the completion.
- We allow MAX_NEW_TOKENS > 8 (e.g., 64) and rely on early stopping for efficiency.
- One question per generation: batch_size = 1 and micro_bs = 1.
- Loader is GPT-OSS + Mxfp4 aware, with attn_implementation="eager",
  and auto offload + max_memory for a 24GB 4090.

"""

import os, re, csv, json, math, time, gc
from typing import List, Tuple, Optional, Dict, Any
from tqdm.auto import tqdm

# PyTorch allocator config
os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:512,expandable_segments:true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

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

MODEL_NAME = "openai/gpt-oss-20b"
OUT_DIR    = "/scratch/pedro.bento/evostar/runs/mmlu/gpt_20b_simple_runs_MMMLU"

# MMLU directory (same used in Evo-Reasoner MCQ script)
BENCH_DIR_MMLU = "/scratch/pedro.bento/evostar/data/MMMLU/test"

TASK_LIMIT   = 100       # cap to first 100 tasks (like Evo script)
RESUME_RUN   = True

# For GPT-OSS we allow some room (64) and rely on early-stop
MAX_NEW_TOKENS = 64
BATCH_SIZE     = 1       # strictly one question per generation
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

# GPT-OSS does NOT support sdpa / flash_attention_2 yet → must use "eager"
if "gpt-oss" in MODEL_NAME:
    ATTN_IMPL = "eager"
else:
    ATTN_IMPL = "flash_attention_2" if _HAS_FLASH2 else "sdpa"

FINAL_TAG = "FINAL_ANSWER:"

# ===================== MCQ extraction helpers =====================

MCQ_LETTER_RE      = re.compile(r"^[A-D]$", re.I)

# More tolerant: match FINAL_ANSWER: <something> anywhere, not only at col 0
our_final_line_pat = re.compile(r"(?i)FINAL_ANSWER\s*:\s*(\S[^\n\r]*)")
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
    # Not used in the main loop, but kept for compatibility if you want it.
    secs = get_sections(text)
    tok = _canon_token(secs.get("ANSWER", "").strip()) or _canon_token(_extract_token_token(text or ""))
    return f"{FINAL_TAG} {tok}" if tok else f"{FINAL_TAG} "


def _strip_trailing_final_lines(text: str) -> str:
    """Remove any trailing lines that contain 'FINAL_ANSWER:' (empty or not)."""
    lines = (text or "").splitlines()
    while lines and re.search(r"(?i)final_answer\s*:", lines[-1]):
        lines.pop()
    return "\n".join(lines)


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


# ===================== Loader (GPT-OSS aware, auto offload) =====================

def _common_model_kwargs(torch_dtype=None, device_map=None, quantization_config=None, max_memory=None):
    """
    Shared kwargs for from_pretrained. For GPT-OSS we rely on its built-in
    Mxfp4Config quantization, so we NEVER pass BitsAndBytesConfig here.
    """
    kw = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        cache_dir=CACHE_DIR,
        attn_implementation=ATTN_IMPL,
    )
    if torch_dtype is not None:
        kw["torch_dtype"] = torch_dtype
    if device_map is not None:
        kw["device_map"] = device_map
    # quantization_config is only used for non-GPT-OSS models with BitsAndBytes;
    # for GPT-OSS (Mxfp4), we leave it as None so HF uses the built-in quantizer.
    if quantization_config is not None:
        kw["quantization_config"] = quantization_config
    if max_memory is not None:
        kw["max_memory"] = max_memory
    return kw


def _try_full_gpu(name, dtype):
    m = AutoModelForCausalLM.from_pretrained(
        name,
        **_common_model_kwargs(torch_dtype=dtype),
    ).to("cuda")
    return m, ("full_gpu_bf16" if dtype == torch.bfloat16
               else "full_gpu_fp16" if dtype == torch.float16
               else "full_gpu_fp32")


def _try_auto_offload(name, dtype, max_memory=None):
    m = AutoModelForCausalLM.from_pretrained(
        name,
        **_common_model_kwargs(torch_dtype=dtype, device_map="auto", max_memory=max_memory),
        offload_folder=OFFLOAD_DIR,
        offload_state_dict=True,
    )
    return m, ("auto_offload_bf16" if dtype == torch.bfloat16
               else "auto_offload_fp16" if dtype == torch.float16
               else "auto_offload_fp32")


def _try_8bit(name):
    """
    BitsAndBytes 8-bit fallback for non-GPT-OSS models.

    GPT-OSS is already quantized with Mxfp4Config and MUST NOT be combined
    with BitsAndBytesConfig, so load_model() skips this for GPT-OSS.
    """
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
    """
    BitsAndBytes 4-bit fallback for non-GPT-OSS models.

    GPT-OSS is already quantized with Mxfp4Config and MUST NOT be combined
    with BitsAndBytesConfig, so load_model() skips this for GPT-OSS.
    """
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

    is_gptoss = "gpt-oss" in name

    # Use global DTYPE for everything (for you: bf16 on 4090)
    dtype_local = DTYPE

    if is_gptoss:
        # GPT-OSS: avoid full-GPU loading; use auto offload with a conservative GPU budget.
        max_mem = None
        if DEVICE == "cuda":
            max_mem = {0: "18GiB", "cpu": "120GiB"}
        try:
            tqdm.write(f"[Loader] Trying auto_offload (GPT-OSS, dtype={dtype_local}, max_memory={max_mem}) ...")
            mdl, mode = _try_auto_offload(name, dtype_local, max_memory=max_mem)
            tqdm.write(f"[Loader]   -> OK ({mode})")
        except Exception as e:
            last_err = e
            tqdm.write(f"[Loader]   -> failed (auto_offload_gptoss): {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        # Non-GPT-OSS: prefer full-GPU bf16 on 4090, then auto offload, then BnB 4/8-bit.
        for loader, label in (
            (_try_full_gpu, "full_gpu"),
            (_try_auto_offload, "auto_offload"),
        ):
            try:
                tqdm.write(f"[Loader] Trying {label} ...")
                mdl, mode = loader(name, dtype_local)
                tqdm.write(f"[Loader]   -> OK ({mode})")
                break
            except Exception as e:
                last_err = e
                tqdm.write(f"[Loader]   -> failed ({label}): {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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


# ===================== Prompt, chat template, stopping =====================

def apply_chat_template(tokenizer, user_text: str) -> str:
    """
    For GPT-OSS, allow remote chat templates if available; otherwise, treat as plain text.
    """
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                add_generation_prompt=True,
                tokenize=False,
            )
    except Exception:
        pass
    return user_text


def build_user_prompt(task_text: str) -> str:
    """
    MCQ-specific prompt: ask for a single FINAL_ANSWER line with a letter.
    """
    return (
        f"{task_text}\n"
        "Choose the correct option based on the question and options above.\n"
        "At the end, output exactly one line:\n"
        "FINAL_ANSWER: <letter>\n"
        "where <letter> is one of: A, B, C, or D.\n"
        "Do not output anything after that line."
    )


class FinalAnswerStopper(StoppingCriteria):
    """
    Stop when a completed 'FINAL_ANSWER: <value>' appears anywhere in the
    generated continuation (after the prompt).
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


# ===================== Generation =====================

def stream_generate(tokenizer, model, tasks: List[str],
                    max_new_tokens: int, batch_size: int):
    """
    Yield: (index, output_text, gen_seconds, out_token_count)

    tasks: list of task_text blocks (as returned by build_mcq_task_block).
    NOTE: We enforce micro_bs = 1 so the model always answers exactly one
    question per forward pass.
    """
    dev = _primary_device(model)
    micro_bs = 1  # strictly one question per time
    max_new = max_new_tokens
    i = 0

    with torch.inference_mode():
        while i < len(tasks):
            batch_tasks = tasks[i:i + micro_bs]
            prompts = [apply_chat_template(tokenizer, build_user_prompt(t)) for t in batch_tasks]
            try:
                t0 = time.time()
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                inputs = {k: v.to(dev) for k, v in inputs.items()}
                start_len = inputs["input_ids"].shape[1]
                stopper = StoppingCriteriaList([
                    FinalAnswerStopper(tokenizer, start_len, tokenizer.eos_token_id, 256)
                ])

                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopper,
                )
                dt = time.time() - t0
                dec = tokenizer.batch_decode(gen, skip_special_tokens=True)

                for j, full in enumerate(dec):
                    pref = prompts[j]
                    # Strip the prompt; keep only what model added after it
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
                # We already use micro_bs=1; only shrink max_new if needed
                if max_new > 8:
                    max_new = max(8, max_new // 2)
                    print(f"[Gen OOM] max_new_tokens -> {max_new}", flush=True)
                    continue
                else:
                    raise


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

    tasks_only = [task_text for (_, task_text, _) in pending]
    gen_iter = stream_generate(tok, mdl, tasks_only, MAX_NEW_TOKENS, BATCH_SIZE)

    total_seen = 0
    total_time = 0.0
    total_out_tokens = 0
    gold_seen = 0
    num_correct = 0
    start = time.time()
    partials_path = os.path.join(run_dir, "partials.jsonl")

    for idx, (tid, task_text, gold_letter) in enumerate(pending):
        _i, ans, dt, out_tok = next(gen_iter)

        # 1) Parse the model token from its raw completion (which includes FINAL_ANSWER: line)
        model_token = _canon_token(_extract_token_token(ans))

        # 2) Sanitize trailing FINAL_ANSWER lines and append exactly one clean line
        sanitized_body = _strip_trailing_final_lines(ans).rstrip()
        if model_token:
            suffix = f"{FINAL_TAG} {model_token}"
        else:
            suffix = f"{FINAL_TAG} "
        if sanitized_body:
            model_answer_raw = sanitized_body + "\n" + suffix
        else:
            model_answer_raw = suffix

        out_tokens = int(out_tok)
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
            "max_new_tokens_start": MAX_NEW_TOKENS,
            "batch_size_start": BATCH_SIZE,
            "gen_seconds": dt,
            "out_tokens": out_tokens,
        }
        with open(os.path.join(run_dir, f"task_{tid}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        total_seen += 1
        total_time += dt
        total_out_tokens += out_tokens

        # gold_seen counts every gold, even when model_token is None (None = wrong)
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
            f"[Task {tid}] FINAL_ANSWER={model_token} | gold={gold_final_token} | gen_seconds={dt:.2f}",
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
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSaved per-task outputs in: {run_dir}\nPartials: {partials_path}", flush=True)


if __name__ == "__main__":
    main()
