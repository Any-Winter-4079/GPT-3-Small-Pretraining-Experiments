# GPT-3 Small Pretraining Experiments

This repo starts coding along [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU), then moves into further improvements ([NanoGPT Speedrun Living Worklog](https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/) and [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)) and own experiments. Every new file adds some new functionality (done this way -vs. commits over the same file- to maintain easier updatability of older versions, in case one day I create a training set for an LLM with `diff`s between consecutive files, e.g., 'Add DDP to this code'. However, currently there are _minor_ changes between certain pairs of files beyond what their name suggests, such as minor comment changes, etc., which we'd need to polish to avoid rewarding the language model e.g., for changes outside 'Add DDP to this code')

## Measurements

- Repo's best time to reach <= 3.28 val loss on 10,485,760 validation tokens: 6.93 minutes ([config](https://huggingface.co/Edue3r4t5y6/gpt-3-small_20251102_154958/blob/main/config.txt), [log](https://huggingface.co/Edue3r4t5y6/gpt-3-small_20251102_154958/raw/main/log.txt)) (4x NVIDIA H100s -we would need to change `total_tokens_per_step_train/val` (262,144), `gpu_batch_size` (8), or `seq_len_train/val` (8,192) to use 8x NVIDIA H100s, but it is a bit costly anyway, about $20/h).
- [Nov 2, 2025, World record <= 3.28 val loss on _the first_ 10,485,760 validation tokens](https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file#world-record-history): 2.345 minutes (8x NVIDIA H100s)

### WR vs This Repo Differences

- As some noticeable differences, this repo tries to be more customizable (better for: learning, customization) while the world record code is (a lot) more optimized (Triton kernels, etc.) but also tries to remove anything that hurts performance (e.g., checkpointing, extra logging, etc.)

## Configuration Options

This repo supports toggling between the following config options:

- FlashAttention (sdpa) / FlexAttention
- Sliding Window Attention (attend to a subset of tokens), Doc Masking (attend to same-doc tokens only), and Attention Logit Soft-capping (if FlexAttention, for performance)
  - Sliding Window Attention ramp (increase window size over training)
  - Attention logit soft-capping ("clamp", "ptx" -faster-, "rational" or "exact")
- Custom masking (e.g., padding mask if non-causal)
- AdamW or AdamW and Muon
  - Muon steps, momentum, use Nesterov
- MHA/MQA/GQA (n_heads vs n_kv_heads)
- QK norm (RMS/L2)
- RMSNorm or LayerNorm
- GELU, ReLU, ReLU\*\*2, SiLU or SwiGLU (fair or unfair) activations
- Bias or no bias
- Tied or untied embeddings
- Learning rate warmup and decay
- RoPE/NoPE/absolute positional encodings
- LM head logit soft-capping
- Gradient norm clipping
- Kernel warmup steps
- ...

## Supports

- Pre-training (stops at val_target)
- Validation (shuffle or always use the same tokens as per speedrun record)
- Sampling (increasingly longer sequences as the model learns)
- Evaluation (HellaSwag one-shot or standard -NOTE: one-shot may require bigger models than 114-124M to work)
- Checkpointing (store checkpoint, resume from checkpoint)
- Logging (to stdout and to file)

## Possible improvements

- FP8 matmul config option (run in FP8 independently for Attention, MLP, LM head)
- Mixture of Experts config option
- Masked Language Model (MLM) training support (`is_causal=False` -> work as encoder)
- Faster validation:

```
new best val loss: 3.30998421
step: 1,850 | val loss: 3.30998421 | val ppl: 27.38 | val time: 9,912.82 ms
...
new best val loss: 3.30172825
step: 1,900 | val loss: 3.30172825 | val ppl: 27.16 | val time: 10,609.94 ms
...
new best val loss: 3.29302740
step: 1,950 | val loss: 3.29302740 | val ppl: 26.92 | val time: 10,428.33 ms
...
new best val loss: 3.28548479
step: 2,000 | val loss: 3.28548479 | val ppl: 26.72 | val time: 8,624.59 ms
```

- Faster shard-switching (100M tokens/shard) time:

```
step: 379 | train loss: 3.87849402 | train ppl: 48.35 | train step time: 174.68 ms | adamw lr: 0.00484023 | tok/s: 1,500,722.21 | total toks: 99,614,720 | total time: 1.07 min | sw size: 512 | max q_scale raw/eff: 1.6959/1.6959 | max k_scale raw/eff: 1.7962/1.7962
step: 380 | train loss: 3.93908834 | train ppl: 51.37 | train step time: 175.82 ms | adamw lr: 0.00483939 | tok/s: 1,490,972.39 | total toks: 99,876,864 | total time: 1.08 min | sw size: 512 | max q_scale raw/eff: 1.6955/1.6955 | max k_scale raw/eff: 1.7964/1.7964
step: 381 | train loss: 3.90186548 | train ppl: 49.49 | train step time: 529.26 ms | adamw lr: 0.00483856 | tok/s: 495,301.83 | total toks: 100,139,008 | total time: 1.08 min | sw size: 512 | max q_scale raw/eff: 1.6958/1.6958 | max k_scale raw/eff: 1.7964/1.7964
step: 382 | train loss: 3.93979096 | train ppl: 51.41 | train step time: 674.42 ms | adamw lr: 0.00483772 | tok/s: 388,695.27 | total toks: 100,401,152 | total time: 1.10 min | sw size: 512 | max q_scale raw/eff: 1.6954/1.6954 | max k_scale raw/eff: 1.7961/1.7961
step: 383 | train loss: 3.94303560 | train ppl: 51.57 | train step time: 185.77 ms | adamw lr: 0.00483688 | tok/s: 1,411,104.72 | total toks: 100,663,296 | total time: 1.10 min | sw size: 512 | max q_scale raw/eff: 1.6958/1.6958 | max k_scale raw/eff: 1.7959/1.7959
step: 384 | train loss: 4.02747297 | train ppl: 56.12 | train step time: 176.83 ms | adamw lr: 0.00483604 | tok/s: 1,482,488.92 | total toks: 100,925,440 | total time: 1.10 min | sw size: 512 | max q_scale raw/eff: 1.6954/1.6954 | max k_scale raw/eff: 1.7954/1.7954
step: 385 | train loss: 3.88525677 | train ppl: 48.68 | train step time: 174.83 ms | adamw lr: 0.00483519 | tok/s: 1,499,416.51 | total toks: 101,187,584 | total time: 1.10 min | sw size: 512 | max q_scale raw/eff: 1.6955/1.6955 | max k_scale raw/eff: 1.7953/1.7953
```

- Reward >1 token at the start of training (e.g., I went `swimming` -> reward `swimming`, `running`, `walking`, etc.) and sharpen signal as training goes on
- ...

## Instructions

I currently use RunPod to run this repo (with other options including: [Lambda](https://lambda.ai/), [Vast.ai](https://vast.ai/)). If you want to run this code on another provider, or your own hardware, feel free to skip RunPod-specific instructions, but stay around for `hf_user` and `hf_token` setup, and the Docker image used.

### 1. Create `hf_user` and `hf_token`

On RunPod, create the following secrets if you want to push your results (and model(s) if checkpointing) to Hugging Face after pre-training.

1. Click `Secrets` (left menu)
2. Click `+ Create Secret`
3. Type `hf_user` as Secret Name
4. Paste or type your Hugging Face username as Secret Value
5. Click Create Secret (to create the secret)
6. Click `+ Create Secret`
7. Type `hf_token` as Secret Name
8. Paste or type your Hugging Face token as Secret Value
9. Click Create Secret (to create the secret)

If you use your own hardware, create them as environment variables

### 2. Choose/create a (RunPod) template, or pull a Docker image

If you use your own hardware, you can:

Use one of the following Docker images (first is the official image, second is a personal backup/clone for my own future sanity, in the rare event RunPod deleted the image, etc.):

- `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- `anywinter4079/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04-runpod-clone`

Or manually match that config (e.g., Python 3.11, CUDA 12.8.1, etc.) on your own system, or try your system as-is (no guarantees)

If you need third-party hardware, you will want a Docker image, some container/volume disk space, and an SSH connection.
On RunPod:

Feel free to create your own template, or [deploy my template](https://console.runpod.io/deploy?template=undhxj45z6&ref=xqdrabty) (Disclaimer: I theoretically get 1% of earnings generated from my template, but the main reason for creating it is to more quickly deploy each time -feel free to create a new one, clone mine, etc.). The template I created uses config:

1. Container Image: anywinter4079/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04-runpod-clone
2. Container Disk: 250 GB
3. Volume Disk: 500 GB
4. Volume Mount Path: /workspace
5. TCP Ports (max 10): Port label: SSH, Port number: 22

If you create your own template, you may want to use RunPod's official container image:

```
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
```

After you have the template (mine or your own), deploy, setting `hf_user` and `hf_token` as Environment variables (note `hf_user` and `hf_token` are added as Environment variables on the deployment page, and not to the template itself, in case the template is made public -e.g., my template is public-). To deploy a template on RunPod:

1. On Deploy a Pod: Choose `H100 XSM` (on `All North America` they seem to be faster/more reliable) and click `Edit Template`
2. Select your GPU count
3. Click `Edit`
4. Click `Environment Variables`
5. Click `+ Add Environment Variable`
6. Click Lock icon to select secret
7. Click `hf_token` to select it as value
8. Type `hf_token` to choose it as key
9. Click `+ Add Environment Variable`
10. Click Lock icon to select secret
11. Click `hf_user` to select it as value
12. Type `hf_user` to choose it as key
13. Click `Set Overrides`
14. Click `Deploy On-Demand`

Finally, connect to the RunPod template via SSH (without SCP & SFTP support), running on your Terminal the provided command, in the following format (replacing `<CONNECTION_STRING>` and `<PRIVATE_KEY_FILE>` with those provided by RunPod):

```
ssh <CONNECTION_STRING>@ssh.runpod.io -i ~/.ssh/<PRIVATE_KEY_FILE>
```

### 3. Install requirements and download `edu_fineweb10B`

To set up the remote machine, before pre-training:

1. Run:

```
cd /workspace && \
apt update && \
apt install -y nano && \
mkdir data && \
cd data && \
nano fineweb-npy.py
```

2. Add `fineweb-npy.py`'s content (from this repo)

3. Save (e.g., `control + x`, `y`, `enter`)

4. Run:

```
python -m pip install --upgrade pip && \
pip install tiktoken datasets tqdm huggingface_hub safetensors && \
python fineweb-npy.py
```

Alternatively, for steps 2 and 3, you could (on RunPod's `workspace`):

```
git clone https://github.com/Any-Winter-4079/GPT-3-Small-Pretraining-Experiments.git
```

to get _all code versions_ into the workspace (if you want to test multiple or do not mind the bloating)

Or clone the repo locally:

```
git clone https://github.com/Any-Winter-4079/GPT-3-Small-Pretraining-Experiments.git
```

`cd` into it:

```
cd GPT-3-Small-Pretraining-Experiments
```

and, on another Terminal window (locally, keeping the other SSH session open), send the files via `scp`, e.g., with (replacing `<PORT>`, `<PRIVATE_KEY_FILE>` and `<IP>` with the values from 'SSH over exposed TCP' on RunPod, and replacing `30-gpt-3-small-with-training-config-and-with-or-without-swa-window-size-ramp.py` with the version you want to send to run):

```
scp -P <PORT> -i ~/.ssh/<PRIVATE_KEY_FILE> -r \
    30-gpt-3-small-with-training-config-and-with-or-without-swa-window-size-ramp.py \
    input.txt \
    data \
    push_to_hub.py \
    root@<IP>:/workspace/
```

## Pre-training

Finally, to pre-train, `cd` out of the `data/` directory (on RunPod):

```
cd ..
```

And either create the file (e.g., `30.py` for brevity) or if you already have it because you have `git clone`'d the repo or `scp`'d the file, go directly to running `torchrun`, specifying your GPU count, and the file name you want to run, e.g.:

```
nano 30.py
```

Add `30-gpt-3-small-with-training-config-and-with-or-without-swa-window-size-ramp.py`'s content (from this repo)

Save (e.g., `control + x`, `y`, `enter`)

And run:

```
torchrun --standalone --nproc_per_node=4 30.py
```

You should see a log similar to:

```
W1102 15:49:55.475000 3485 torch/distributed/run.py:766]
W1102 15:49:55.475000 3485 torch/distributed/run.py:766] *****************************************
W1102 15:49:55.475000 3485 torch/distributed/run.py:766] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W1102 15:49:55.475000 3485 torch/distributed/run.py:766] *****************************************
per-gpu gradient accumulation mini-steps: 1
lr warmup steps: 0
lr warmup and cosine steps: 3,051
max train steps: 19,073
10,485,760 val tokens to be consumed in 40 steps (262,144 tokens per val step)
num decayed parameter tensors (AdamW): 25, with 38,633,664 parameters
num non-decayed parameter tensors (AdamW): 25, with 19,200 parameters
num Muon parameter tensors: 72, with 75,497,472 parameters
using fused AdamW: True
found 99 shards for split train
README.md: 7.02kB [00:00, 15.1MB/s]
data/train-00000-of-00001.parquet: 100%|███████████████████████████████████████████████████████████████████████████████| 24.4M/24.4M [00:00<00:00, 33.8MB/s]
data/test-00000-of-00001.parquet: 100%|████████████████████████████████████████████████████████████████████████████████| 6.11M/6.11M [00:00<00:00, 19.6MB/s]
data/validation-00000-of-00001.parquet: 100%|██████████████████████████████████████████████████████████████████████████| 6.32M/6.32M [00:00<00:00, 23.5MB/s]
Generating train split: 100%|██████████████████████████████████████████████████████████████████████████████| 39905/39905 [00:00<00:00, 301180.80 examples/s]
Generating test split: 100%|███████████████████████████████████████████████████████████████████████████████| 10003/10003 [00:00<00:00, 328932.14 examples/s]
Generating validation split: 100%|█████████████████████████████████████████████████████████████████████████| 10042/10042 [00:00<00:00, 311047.11 examples/s]
total train shard 0 tokens: 99,876,865
total gpus: 4
gpu batch size: 8 sequences
sequence length: 8,192 tokens
tokens fed to gpu per grad accum mini-step: 65,536 (4 gpus, 262,144 total tokens)
per-gpu grad accumulation mini-steps for train shard 0 (each mini-step processing 262,144 tokens): 381
found 1 shards for split val
[rank 2] gets HellaSwag sentences 5,022 to 7,532
[rank 3] gets HellaSwag sentences 7,533 to 10,041
total val shard 0 tokens: 99,876,865
total gpus: 4
gpu batch size: 8 sequences
sequence length: 8,192 tokens
tokens fed to gpu per grad accum mini-step: 65,536 (4 gpus, 262,144 total tokens)
per-gpu grad accumulation mini-steps for val shard 0 (each mini-step processing 262,144 tokens): 381
[rank 1] gets HellaSwag sentences 2,511 to 5,021
[rank 0] gets HellaSwag sentences 0 to 2,510
training and model configs saved to ./configs_and_logs/20251102_154958/config.txt
114,150,336 parameters
```

Then the code takes 2-4 minutes to compile, and then:

```
step: 0 | train loss: 10.98035240 | train ppl: 58,709.24 | train step time: 171.81 ms | adamw lr: 0.00500000 | tok/s: 1,525,741.15 | total toks: 262,144 | total time: 0.00 min | sw size: 256 | max q_scale raw/eff: 1.0045/1.0045 | max k_scale raw/eff: 1.0045/1.0045
step: 1 | train loss: 10.25063896 | train ppl: 28,300.62 | train step time: 166.76 ms | adamw lr: 0.00500000 | tok/s: 1,571,963.77 | total toks: 524,288 | total time: 0.01 min | sw size: 256 | max q_scale raw/eff: 1.0090/1.0090 | max k_scale raw/eff: 1.0090/1.0090
step: 2 | train loss: 8.28910255 | train ppl: 3,980.26 | train step time: 166.57 ms | adamw lr: 0.00500000 | tok/s: 1,573,775.03 | total toks: 786,432 | total time: 0.01 min | sw size: 256 | max q_scale raw/eff: 1.0130/1.0130 | max k_scale raw/eff: 1.0122/1.0122
step: 3 | train loss: 8.08094215 | train ppl: 3,232.28 | train step time: 167.56 ms | adamw lr: 0.00499999 | tok/s: 1,564,477.46 | total toks: 1,048,576 | total time: 0.01 min | sw size: 256 | max q_scale raw/eff: 1.0166/1.0166 | max k_scale raw/eff: 1.0154/1.0154
step: 4 | train loss: 8.21518803 | train ppl: 3,696.67 | train step time: 165.34 ms | adamw lr: 0.00499998 | tok/s: 1,585,436.96 | total toks: 1,310,720 | total time: 0.01 min | sw size: 256 | max q_scale raw/eff: 1.0196/1.0196 | max k_scale raw/eff: 1.0184/1.0184
...
```

## Pushing the logs and checkpoints to Hugging Face

To push the config, log, and model(s) (if checkpointing), find the latest timestamp, either on `checkpoints` or `configs_and_logs`:

```
cd checkpoints
```

Copy the timestamp

```
cd ..
```

Create `push_to_hub.py` (adding its content from this repo) and/or edit it (e.g., if you have `scp`'d the script):

```
nano push_to_hub.py
```

Replacing the `timestamp` `("20251013_145053")` with your timestamp (**TODO**: make the script ask for the timestamp as input when running)

Save (e.g., `control + x`, `y`, `enter`)

And run:

```
python push_to_hub.py
```

## Storing `stdout` and `stderr` on file (if you ever need to)

Because error messages can be **very** long, you can run (replacing `--nproc_per_node=1` with you GPU count):

```
mkdir debug
```

```
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=1 30.py \
  > >(stdbuf -oL tee -a debug/out-$(date +%F_%H-%M-%S).log) \
  2> >(stdbuf -oL tee -a debug/err-$(date +%F_%H-%M-%S).log >&2)
```

Then review the error message in chunks (replacing `err-2025-10-25_00-32-19.log` with your own log):

```
cd debug
awk 'NR>=0 && NR<=180' err-2025-10-25_00-32-19.log
```
