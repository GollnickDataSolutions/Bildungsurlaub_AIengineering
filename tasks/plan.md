# AI Engineering Bildungsurlaub — Work Plan

## Context

Five-day course covering the full ML/AI engineering stack: PyTorch fundamentals →
Computer Vision → LLMs → Deployment. Course material lives in `D01`–`D05` folders.
Exercises follow a `_start.py` (stub) / `_end.py` (reference solution) pattern.

No SPEC.md exists; this plan is derived directly from the course structure.

---

## Dependency Graph

```
PyTorch Track
─────────────
D01 (Foundations)
  └─► D02 (Classification)
        └─► D03 (Computer Vision)

LangChain / LLM Track  (independent of PyTorch track)
──────────────────────
D04 (LLMs + Chains)
  └─► D05 (Prompt Eng. + Deployment)

Merge point
───────────
D05 brings both tracks together:
  - Autoencoders  ←  PyTorch track
  - Streamlit app ←  LangChain track

Bonus (outside course scope)
─────────────────────────────
torch_test.py  ←  Diffusers / FLUX image generation
  depends on: CUDA GPU, HuggingFace login, bug fix in seed line
```

---

## Phase 1 — PyTorch Foundations (D01)

**Goal:** build intuition for tensors, autograd, and the PyTorch training loop from
scratch before using `nn.Module`.

### Task 1.1 — Neural Network from scratch
- **File:** `D01_Intro/010__NN_from_scratch/nn_scratch_start.py`
- **Reference:** `nn_scratch_end.py`
- Implement forward pass, loss, and weight updates using only numpy/raw Python.
- **Acceptance criteria:** script runs end-to-end; loss decreases over epochs.
- **Verification:** compare final accuracy against `nn_scratch_end.py`.

### Task 1.2 — Linear Regression progression (6 exercises)
Each builds on the previous; complete in numeric order.

| # | File | Topic |
|---|------|-------|
| a | `00_LinRegFromScratch_start.py` | Manual gradient descent, no `nn` |
| b | `10_LinReg_ModelClass_start.py` | Introduce `nn.Module` |
| c | `20_LinReg_Batches_start.py` | Mini-batch training loop |
| d | `30_LinReg_DatasetDataloader_start.py` | `Dataset` + `DataLoader` |
| e | `40_LinReg_ModelSavingLoading_start.py` | `torch.save` / `torch.load` |
| f | `50_LinReg_HyperparameterTuning_start.py` | Grid search over LR / epochs |

- **Acceptance criteria:** each script produces matching loss curve vs. `_end.py`.
- **Verification:** run both versions, compare final loss values.

**Checkpoint 1:** after Task 1.2f, training pipeline is fully understood.

---

## Phase 2 — Classification (D02)

**Goal:** extend the training loop to classification problems; meet PyTorch Lightning.

**Depends on:** Phase 1 (especially Task 1.2b — `nn.Module` pattern).

### Task 2.1 — Multi-class classification
- **File:** `D02_Classification/010_Classification/MultiClassClassification_start.py`
- **Reference:** `MultiClassClassification_end.py`
- Adapt linear model to softmax + cross-entropy loss; use `iris_classification.py` as comparison.
- **Acceptance criteria:** accuracy > 80 % on test set.

### Task 2.2 — Multi-label classification
- **File:** `D02_Classification/010_Classification/MultilabelClassification_start.py`
- **Reference:** `MultilabelClassification_end.py`
- Multiple sigmoid outputs; use BCEWithLogitsLoss.
- **Acceptance criteria:** F1 macro > 0.70.

### Task 2.3 — PyTorch Lightning refactor
- **File:** `D02_Classification/30_PyTorchLightning/lightning_intro_end.py` (study only)
- Understand `LightningModule`, `Trainer`, automatic GPU/logging.
- **Acceptance criteria:** can explain what Lightning replaces manually.

**Checkpoint 2:** full classification pipeline works in raw PyTorch and Lightning.

---

## Phase 3 — Computer Vision (D03)

**Goal:** apply CNNs to real image datasets; understand transfer learning.

**Depends on:** Phase 2 (classification loop, `DataLoader` with `ImageFolder`).

### Task 3.1 — Layer calculations
- **File:** `D03_ComputerVision/LayerCalculations_start.py`
- Calculate output dimensions of Conv2d + MaxPool2d by hand, then verify in code.
- **Acceptance criteria:** formula matches PyTorch output shapes.

### Task 3.2 — Image preprocessing
- **File:** `D03_ComputerVision/ImagePreprocessing_start.py`
- Practice `torchvision.transforms` pipeline: resize, normalize, augment.

### Task 3.3 — CNN Binary Classification
- **File:** `D03_ComputerVision/BinaryClassification/CNN_BinaryClassification_start.py`
- Already largely filled in; focus on GPU utilization and confusion matrix.
- **Acceptance criteria:** accuracy > 90 %.

### Task 3.4 — CNN Multiclass Classification
- **File:** `D03_ComputerVision/MulticlassClassification/Cnn_MulticlassClassification_start.py`
- **Reference:** `Cnn_MulticlassClassification_end.py`
- **Acceptance criteria:** accuracy > 80 %, confusion matrix plotted.

### Task 3.5 — Transfer Learning / Fine-tuning
- **File:** `D03_ComputerVision/ModelFinetuning/TransferLearning_start.py`
- **Reference:** `TransferLearning_end.py`
- Freeze backbone, replace head, unfreeze selectively.
- **Acceptance criteria:** fine-tuned model outperforms training from scratch.

### Task 3.6 — Audio Classification (capstone)
- **Files:** `D03_ComputerVision/AudioClassification/` — `data_prep.py`, `eda.py`, `modeling.py`
- End-to-end pipeline on audio data; uses spectrogram as image.
- **Acceptance criteria:** model runs without errors; EDA plots generated.

### Task 3.7 — Object Detection (exploration)
- **File:** `D03_ComputerVision/ObjectDetection/ObjectDetection.py`
- Run and understand; no modifications needed.

**Checkpoint 3:** full CV pipeline from raw images to fine-tuned model works on GPU.

---

## Phase 4 — LLMs & LangChain (D04)

**Goal:** build chains with LangChain using OpenAI, Groq, and Ollama backends.

**Depends on:** nothing from PyTorch track; needs API keys in `.env`.

### Task 4.1 — Basic chat (OpenAI + Groq + Ollama)
- **Files:** `10_model_chat_openai.py`, `20_model_chat_groq.py`, `20_model_chat_ollama.py`
- Run each; confirm working API keys / local Ollama instance.

### Task 4.2 — Prompt Templates
- **File:** `30_prompt_templates.py`
- Practice `ChatPromptTemplate`, system/user separation.

### Task 4.3 — Simple chain
- **File:** `40_simple_chain.py`
- Chain = prompt | model | parser. Study LCEL pipe operator.

### Task 4.4 — Parallel chain
- **File:** `50_parallel_chain.py`
- `RunnableParallel` for branching outputs.

### Task 4.5 — Multimodal
- **File:** `60_multimodal.py`
- Send images to vision model; understand base64 encoding.

### Task 4.6 — Structured Outputs (email sorter)
- **File:** `structured_outputs/structured_outputs.py`
- `PydanticOutputParser` + Enum classes + batch processing from JSON files.
- **Acceptance criteria:** all emails in `krankenkassen_emails_dataset/` get a category + complexity.

**Checkpoint 4:** LangChain LCEL pipeline works end-to-end with multiple backends.

---

## Phase 5 — Prompt Engineering & Deployment (D05)

**Goal:** prompt engineering techniques + ship a working Streamlit app.

**Depends on:** Phase 4 (LangChain), Phase 2 (PyTorch autoencoders).

### Task 5.1 — Self-Consistency CoT
- **File:** `D05_PromptEng_Deployment/self_consistency_cot.py`
- Already complete; run it, understand majority-voting over multiple model calls.

### Task 5.2 — Autoencoders
- **File:** `D05_PromptEng_Deployment/Autoencoders/Autoencoders_end.py`
- Encoder–decoder architecture for apple/banana images.
- **Acceptance criteria:** reconstruction loss < 0.05; reconstructed images recognizable.

### Task 5.3 — Streamlit deployment
- **File:** `D05_PromptEng_Deployment/app/app.py`
- Launch `streamlit run app.py`; interact with the Nostradamus chatbot.
- Extend: add streaming output (`stream=True`) or chat history.
- **Acceptance criteria:** app runs in browser, returns LLM response.

**Checkpoint 5:** deployed app works end-to-end in browser.

---

## Bonus — FLUX Image Generation

**Goal:** generate an image locally using `diffusers` + `FLUX.1-schnell` on GPU.

**Depends on:** CUDA GPU available (torch_test.py line 3 confirms it), HuggingFace login.

### Task B.1 — Fix torch_test.py
- **Bug:** line 20 uses `torch.Generator("cuda").manual_seed=42` (assignment instead of call).
- **Fix:** `torch.Generator("cuda").manual_seed(42)`.
- Remove the `if False else` ternary — it's dead code.
- **Acceptance criteria:** `output.png` is created with the Border Collie image.

### Task B.2 — Explore generation parameters
- Try different `num_inference_steps` (1, 2, 4, 8) and compare quality.
- Try different image sizes (512×512 vs 1024×1024) and measure GPU memory.

---

## Risk & Notes

| Risk | Mitigation |
|------|-----------|
| HuggingFace FLUX download (~24 GB) takes time | Run overnight; model cached in `~/.cache/huggingface` |
| `.env` file missing API keys | Create from `.env.example` or add OPENAI_API_KEY, GROQ_API_KEY |
| Ollama not running locally | `ollama serve` + `ollama pull llama3` before D04 |
| `krankenkassen_emails_dataset/` not in repo | Check if data ships separately with course materials |
