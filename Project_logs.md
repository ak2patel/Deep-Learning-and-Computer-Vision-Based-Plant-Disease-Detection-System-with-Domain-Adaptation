# 📔 AgriVision: Development Journal

This journal tracks the technical journey, architectural decisions, and performance benchmarks for the AgriVision 2.0 Major Project.

---

## 🧠 Part 1: Architectural Decisions & "The Why"

### 1. Dataset Strategy: "The Data Diet"
- **The Problem**: Minor project used only PlantVillage (Lab images). Failed in real field conditions (sun/shadows).
- **The Decision**: Integrated **PlantDoc (Field Images)** to teach the model "Domain Adaptation"—ignoring background noise.
- **Mental Model**: PlantVillage teaches the *disease*; PlantDoc teaches the *environment*.

### 2. Custom Architecture: "AgriNet"
- **The Problem**: Standard models (ResNet/EfficientNet) are complex for beginners and too heavy for mobile.
- **The Decision**: Designed **AgriNet** using **Depthwise Separable Convolutions** (Inverted Residual Blocks).
- **Learning Outcome**: Split the spatial filtering from the color mixing to achieve high speed without losing accuracy.

### 3. Training Optimization: "The Gym"
- **Augmentations**: Used Random Flips and Color Jitter to prevent **Overfitting** (memorizing the data).
- **Optimizer**: Used **Adam** for adaptive learning.
- **Scheduler**: Used **ReduceLROnPlateau** to automatically slow down learning when nearing peak accuracy.

---

## 📊 Part 2: System Environment & Benchmarks

### 🖥️ Hardware Profile
- **PyTorch Version**: 2.5.1+cu121
- **CUDA**: Enabled (Training on GPU)
- **GPU**: NVIDIA GeForce GTX 1650 (4GB VRAM)

### 📈 Training Performance
- **Training Samples**: 43,444 images
- **Validation Samples**: 10,861 images
- **Classes**: 38 Disease Categories
- **Avg. Epoch Time**: ~12.5 minutes

---

## 🚀 Part 3: Live Training Logs (Mar 23, 2026)
Epoch 1/10
--------------------
Training: 100%|█████████████████| 2716/2716 [12:32<00:00,  3.61it/s, loss=2.48]
Train Loss: 0.8001 Acc: 0.7635
Validation: 100%|████████████████████████████| 679/679 [01:23<00:00,  8.16it/s] 
Val Loss: 0.3566 Acc: 0.8908
==> Model saved to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_v1.pth (improved val_loss)

Epoch 2/10
--------------------
Training: 100%|████████████████| 2716/2716 [11:27<00:00,  3.95it/s, loss=0.367] 
Train Loss: 0.2985 Acc: 0.9046
Validation: 100%|████████████████████████████| 679/679 [01:09<00:00,  9.83it/s] 
Val Loss: 0.1645 Acc: 0.9468
==> Model saved to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_v1.pth (improved val_loss)

Epoch 3/10
--------------------
Training: 100%|███████████████| 2716/2716 [11:46<00:00,  3.84it/s, loss=0.0218] 
Train Loss: 0.2098 Acc: 0.9330
Validation: 100%|████████████████████████████| 679/679 [01:04<00:00, 10.48it/s] 
Val Loss: 0.1371 Acc: 0.9551
==> Model saved to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_v1.pth (improved val_loss)

Epoch 4/10
--------------------
Training: 100%|████████████████| 2716/2716 [11:20<00:00,  3.99it/s, loss=0.114] 
Train Loss: 0.1691 Acc: 0.9457
Validation: 100%|████████████████████████████| 679/679 [01:06<00:00, 10.20it/s] 
Val Loss: 0.1120 Acc: 0.9627
==> Model saved to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_v1.pth (improved val_loss)

Epoch 5/10
--------------------
Training: 100%|████████████████| 2716/2716 [11:18<00:00,  4.00it/s, loss=0.483] 
Train Loss: 0.1445 Acc: 0.9533
Validation: 100%|████████████████████████████| 679/679 [01:05<00:00, 10.38it/s] 
Val Loss: 0.0783 Acc: 0.9750
==> Model saved to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_v1.pth (improved val_loss)

Epoch 6/10
--------------------
Training: 100%|█████████████████| 2716/2716 [11:17<00:00,  4.01it/s, loss=0.17] 
Train Loss: 0.1214 Acc: 0.9609
Validation: 100%|████████████████████████████| 679/679 [01:04<00:00, 10.55it/s] 
Val Loss: 0.0636 Acc: 0.9781
==> Model saved to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_v1.pth (improved val_loss)

Epoch 7/10
--------------------
Training: 100%|█████████████| 2716/2716 [11:17<00:00,  4.01it/s, loss=0.000853] 
Train Loss: 0.1120 Acc: 0.9642
Validation: 100%|████████████████████████████| 679/679 [07:01<00:00,  1.61it/s] 
Val Loss: 0.0883 Acc: 0.9725

Epoch 8/10
--------------------
Training: 100%|████████████████| 2716/2716 [55:02<00:00,  1.22s/it, loss=0.165] 
Train Loss: 0.1044 Acc: 0.9657
Validation: 100%|████████████████████████████| 679/679 [01:11<00:00,  9.54it/s]
Val Loss: 0.0513 Acc: 0.9823
==> Model saved to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_v1.pth (improved val_loss)

Epoch 9/10
--------------------
Training: 100%|████████████████| 2716/2716 [11:29<00:00,  3.94it/s, loss=0.357] 
Train Loss: 0.0939 Acc: 0.9698
Validation: 100%|████████████████████████████| 679/679 [01:09<00:00,  9.79it/s] 
Val Loss: 0.0964 Acc: 0.9685

Epoch 10/10
--------------------
Training: 100%|███████████████| 2716/2716 [12:48<00:00,  3.54it/s, loss=0.0388] 
Train Loss: 0.0853 Acc: 0.9721
Validation: 100%|████████████████████████████| 679/679 [01:29<00:00,  7.62it/s] 
Val Loss: 0.0396 Acc: 0.9875
==> Model saved to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_v1.pth (improved val_loss)

Training Complete. Best validation loss: 0.0396
Model saved at: D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_v1.pth
PS D:\Dev\AI-Powered-Plant-Disease-Detection-System> 
---
Starting MIXED DOMAIN Fine-Tuning on cuda
Skipping unknown PlantDoc class: Tomato mold leaf
Loaded 54305 PlantVillage base images.
Loaded 2231 PlantDoc mapped field images.
Loading Base Weights: D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_v1.pth
D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\mixed_finetune_agrinet.py:179: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=device))      

Epoch 1/10
--------------------
Training: 100%|████████████████| 2827/2827 [11:44<00:00,  4.01it/s, loss=0.382]
Train Loss: 0.5123 Acc: 0.9070
Validation: 100%|████████████████████████████| 707/707 [01:44<00:00,  6.80it/s]
Val Loss: 0.3257 Acc: 0.9383
==> Mixed Domain Adaptation successful. Saved robust model to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_mixed_v1.pth

Epoch 2/10
--------------------
Training: 100%|███████████████| 2827/2827 [11:26<00:00,  4.12it/s, loss=0.0315] 
Train Loss: 0.3337 Acc: 0.9269
Validation: 100%|████████████████████████████| 707/707 [01:24<00:00,  8.38it/s] 
Val Loss: 0.2591 Acc: 0.9459
==> Mixed Domain Adaptation successful. Saved robust model to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_mixed_v1.pth

Epoch 3/10
--------------------
Training: 100%|███████████████| 2827/2827 [11:46<00:00,  4.00it/s, loss=0.0826] 
Train Loss: 0.2831 Acc: 0.9348
Validation: 100%|██████████████████████████| 707/707 [1:37:42<00:00,  8.29s/it] 
Val Loss: 0.2231 Acc: 0.9519
==> Mixed Domain Adaptation successful. Saved robust model to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_mixed_v1.pth

Epoch 4/10
--------------------
Training: 100%|████████████████| 2827/2827 [11:24<00:00,  4.13it/s, loss=0.204]
Train Loss: 0.2525 Acc: 0.9389
Validation: 100%|████████████████████████████| 707/707 [02:08<00:00,  5.50it/s] 
Val Loss: 0.2065 Acc: 0.9530
==> Mixed Domain Adaptation successful. Saved robust model to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_mixed_v1.pth

Epoch 5/10
--------------------
Training: 100%|████████████████| 2827/2827 [11:19<00:00,  4.16it/s, loss=0.349] 
Train Loss: 0.2349 Acc: 0.9414
Validation: 100%|████████████████████████████| 707/707 [01:26<00:00,  8.14it/s] 
Val Loss: 0.1838 Acc: 0.9568
==> Mixed Domain Adaptation successful. Saved robust model to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_mixed_v1.pth

Epoch 6/10
--------------------
Training: 100%|████████████████| 2827/2827 [11:18<00:00,  4.17it/s, loss=0.285] 
Train Loss: 0.2247 Acc: 0.9434
Validation: 100%|████████████████████████████| 707/707 [01:28<00:00,  7.99it/s] 
Val Loss: 0.1782 Acc: 0.9595
==> Mixed Domain Adaptation successful. Saved robust model to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_mixed_v1.pth

Epoch 7/10
--------------------
Training: 100%|███████████████| 2827/2827 [13:01<00:00,  3.62it/s, loss=0.0199] 
Train Loss: 0.2101 Acc: 0.9462
Validation: 100%|████████████████████████████| 707/707 [02:28<00:00,  4.75it/s] 
Val Loss: 0.1774 Acc: 0.9598
==> Mixed Domain Adaptation successful. Saved robust model to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_mixed_v1.pth

Epoch 8/10
--------------------
Training: 100%|███████████████| 2827/2827 [11:35<00:00,  4.07it/s, loss=0.0148] 
Train Loss: 0.2035 Acc: 0.9467
Validation: 100%|████████████████████████████| 707/707 [01:47<00:00,  6.57it/s] 
Val Loss: 0.1778 Acc: 0.9571

Epoch 9/10
--------------------
Training: 100%|█████████████████| 2827/2827 [11:44<00:00,  4.02it/s, loss=0.16] 
Train Loss: 0.1936 Acc: 0.9499
Validation: 100%|████████████████████████████| 707/707 [01:45<00:00,  6.70it/s] 
Val Loss: 0.1587 Acc: 0.9614
==> Mixed Domain Adaptation successful. Saved robust model to D:\Dev\AI-Powered-Plant-Disease-Detection-System\ml-training\models\agrinet_mixed_v1.pth

Epoch 10/10
--------------------
Training: 100%|████████████████| 2827/2827 [12:25<00:00,  3.79it/s, loss=0.609] 
Train Loss: 0.1877 Acc: 0.9496
Validation: 100%|████████████████████████████| 707/707 [01:39<00:00,  7.12it/s] 
Val Loss: 0.1652 Acc: 0.9601
PS D:\Dev\AI-Powered-Plant-Disease-Detection-System> 



## 🔮 Part 4: Real RAG Implementation (May 23, 2026)
Successfully completed the RAG (Retrieval-Augmented Generation) engine integration for grounded agricultural advice.

### Implementation Details:
- **PDF Ingestion & Chunking**: Integrated a pipeline to load PDF manuals (Apple, Potato, Tomato, Maize) from `rag-engine/documents/`, splitting text into chunks (1000 characters, 200 overlap) using `RecursiveCharacterTextSplitter`.
- **Vector Embeddings**: Used `sentence-transformers/all-MiniLM-L6-v2` locally on CPU to generate 384-dimensional dense vectors for 125 document chunks.
- **Vector Database**: Used `FAISS` to build a local vector index saved under `rag-engine/faiss_index/`.
- **LLM Synthesis**: Integrated the Groq API utilizing the `llama-3.1-8b-instant` model for high-speed, grounded generation.
- **Strict Grounding Prompt**: Configured a system prompt that restricts the model from hallucinating or answering from external knowledge, ensuring all advice is grounded strictly in the provided agricultural PDF manuals.

### Verified Behavior:
- **Safety Grounding**: Querying vague symptoms (e.g., general leaf yellowing) correctly yields a grounded refusal to guess without context.
- **Accurate Retrieval**: Querying specific conditions (e.g., tomato wilt) successfully retrieves corresponding pages of `tomato.pdf` and extracts symptoms and treatments (Bacterial vs. Fusarium wilt) with no hallucinations.

## 📝 Part 5: B.Tech Major Project Report Drafting (May 23, 2026)
- **Activity**: Created the comprehensive report draft `report_content.txt` at the workspace root.
- **Details**:
  - The document is structured strictly according to the format specified in `B. Tech. Major Project Report Format.docx`.
  - It compiles complete texts, tables, mathematical formulations (e.g., Depthwise Separable Convolutions, metrics, and Grad-CAM), database schemas, DFD diagrams, and IEEE references matching the actual AgriVision 2.0 implementation.
  - The entire report content was written using the passive voice / third-person style to maintain academic and professional objectivity as mandated.
