# Task List — AI Engineering Bildungsurlaub

## Phase 1 — PyTorch Foundations (D01)

- [ ] **1.1** Complete `nn_scratch_start.py` — NN forward/backward without `nn.Module`
- [ ] **1.2a** Complete `00_LinRegFromScratch_start.py` — manual gradient descent
- [ ] **1.2b** Complete `10_LinReg_ModelClass_start.py` — introduce `nn.Module`
- [ ] **1.2c** Complete `20_LinReg_Batches_start.py` — mini-batch training loop
- [ ] **1.2d** Complete `30_LinReg_DatasetDataloader_start.py` — `Dataset` + `DataLoader`
- [ ] **1.2e** Complete `40_LinReg_ModelSavingLoading_start.py` — save/load model weights
- [ ] **1.2f** Complete `50_LinReg_HyperparameterTuning_start.py` — grid search
- [ ] **CHECKPOINT 1** ✓ training pipeline understood

## Phase 2 — Classification (D02)

- [ ] **2.1** Complete `MultiClassClassification_start.py` — softmax + cross-entropy
- [ ] **2.2** Complete `MultilabelClassification_start.py` — sigmoid + BCEWithLogitsLoss
- [ ] **2.3** Study `lightning_intro_end.py` — understand PyTorch Lightning abstractions
- [ ] **CHECKPOINT 2** ✓ classification works in raw PyTorch + Lightning

## Phase 3 — Computer Vision (D03)

- [ ] **3.1** Complete `LayerCalculations_start.py` — Conv2d output shape formulas
- [ ] **3.2** Complete `ImagePreprocessing_start.py` — torchvision transforms pipeline
- [ ] **3.3** Run + review `CNN_BinaryClassification_start.py` (mostly filled in)
- [ ] **3.4** Complete `Cnn_MulticlassClassification_start.py`
- [ ] **3.5** Complete `TransferLearning_start.py` — freeze backbone, replace head
- [ ] **3.6** Run audio classification pipeline (`data_prep.py` → `eda.py` → `modeling.py`)
- [ ] **3.7** Run `ObjectDetection.py` — exploration only
- [ ] **CHECKPOINT 3** ✓ full CV pipeline works on GPU

## Phase 4 — LLMs & LangChain (D04)

- [ ] **4.1** Run `10_model_chat_openai.py` — verify OPENAI_API_KEY works
- [ ] **4.1b** Run `20_model_chat_groq.py` — verify GROQ_API_KEY works
- [ ] **4.1c** Run `20_model_chat_ollama.py` — verify Ollama local instance works
- [ ] **4.2** Study + run `30_prompt_templates.py`
- [ ] **4.3** Study + run `40_simple_chain.py` — LCEL pipe operator
- [ ] **4.4** Study + run `50_parallel_chain.py` — `RunnableParallel`
- [ ] **4.5** Study + run `60_multimodal.py` — image inputs to vision model
- [ ] **4.6** Run `structured_outputs.py` on all emails in `krankenkassen_emails_dataset/`
- [ ] **CHECKPOINT 4** ✓ LangChain LCEL pipeline works end-to-end

## Phase 5 — Prompt Engineering & Deployment (D05)

- [ ] **5.1** Run `self_consistency_cot.py` — understand majority-voting CoT
- [ ] **5.2** Run `Autoencoders_end.py` — train autoencoder on apple/banana images
- [ ] **5.3** Launch `streamlit run D05_PromptEng_Deployment/app/app.py` — test in browser
- [ ] **5.3+** Extend app: add streaming or chat history
- [ ] **CHECKPOINT 5** ✓ app deployed and running in browser

## Bonus — FLUX Image Generation

- [ ] **B.1** Fix `torch_test.py` line 20: `manual_seed=42` → `manual_seed(42)`
- [ ] **B.1b** Remove dead `if False else` ternary on same line
- [ ] **B.2** Run fixed `torch_test.py` — verify `output.png` is created
- [ ] **B.3** Experiment with `num_inference_steps` (1 / 2 / 4 / 8) and compare quality
