# 🌿 AI powered plant disease detection system

An advanced, end-to-end Machine Learning pipeline and "Real RAG" application for agricultural disease detection. This system is designed as an academic Major Project to solve the "Lab vs. Field" domain adaptation problem using a custom Convolutional Neural Network (CNN) architecture.

## 👥 Project Team & Roles

| Name | Roll Number | Primary Responsibility |
| :--- | :--- | :--- |
| **Aman Kumar Bind** | `2201086CS` | **ML Architect** (Custom CNN, Mixed-Domain Training) |
| **Ankit Kumar Patel** | `2201116CS` | **Backend Engineer** (FastAPI, PostgreSQL, Auth Flow) |
| **Sahil Morwal** | `2201085CS` | **AI Integration Lead** (LangChain, FAISS, Real RAG System) |
| **Nawazish Hassan** | `2201031CS` | **Frontend & UI Architect** (Next.js, Glassmorphism, PDF Reports) |

## 🏆 Project Achievements (What is Done)

### 1. Custom CNN Architecture ("AgriNet")
Instead of using off-the-shelf models like ResNet or EfficientNet, we built a custom architecture from scratch using **Depthwise Separable Convolutions** and **Inverted Residual Blocks**.
- **Result:** Blazing fast inference speeds (perfect for mobile farming apps) while maintaining top-tier accuracy.

### 2. High-Accuracy Base Model
- Trained on **PlantVillage** (54,000+ clean lab images).
- Achieved **98.75% Validation Accuracy** across 38 distinct plant/disease classes.
- Used Adam optimizer and ReduceLROnPlateau scheduling to perfectly navigate the loss landscape.

### 3. "Mixed-Domain" Fine-Tuning (Real-World Robustness)
Laboratory conditions (PlantVillage) fail in the real world due to shadows, dirt, and messy backgrounds. To solve this, we used **Domain Adaptation**.
- Mapped 27 real-world field classes from the **PlantDoc** dataset to our standard 38 classes.
- Implemented `mixed_finetune_agrinet.py` to jointly train the base model on both clean lab data and messy field data simultaneously.
- **Pivotal Result:** Achieved **96.1% Validation Accuracy** in mixed-domain testing. 
- *The trained models are located in `ml-training/models/` and are fully tracked.*
- *The immense 3GB+ image datasets have been appropriately `.gitignore`d to maintain a clean repository.*

### 4. Knowledge-Grounded AI ("Real RAG Engine")
Instead of generic, hallucination-prone chatbots, we built a fully local Retrieval-Augmented Generation (RAG) engine.
- **Pipeline:** Parses PDF manuals (`apple.pdf`, `potato.pdf`, `tomato.pdf`, `maize.pdf`), chunks text via `RecursiveCharacterTextSplitter`, and embeds them locally on CPU using `sentence-transformers/all-MiniLM-L6-v2`.
- **Database:** Stores vectors in a local `FAISS` database for sub-millisecond similarity search.
- **Generation:** Queries Groq API (`llama-3.1-8b-instant`) for fast, context-grounded synthesis.
- **Result:** Provides farmers with treatment recommendations verified directly from agricultural literature, refusing to answer if the context does not cover the disease.

## 🚀 Roadmap (What is Left)

The Machine Learning architecture is officially complete and proven. We are now moving into the Full-Stack and Intelligence phases.

### 1. ⚙️ Core Backend (FastAPI + PostgreSQL)
- **Objective**: Move off local memory to a robust database.
- **Features**: JWT User Authentication, user profiles (`users` table), and prediction history logging (`predictions` table).

### 2. 📚 FastAPI RAG Endpoint Integration
- **Objective**: Hook the completed RAG engine to Ankit's FastAPI server.
- **Features**: Expose a POST `/chat` API endpoint that receives queries from the frontend, queries the FAISS index, and returns the grounded answer to the client.

### 3. ✨ Frontend Redesign (Premium Glassmorphism)
- **Objective**: Upgrade the UI/UX from a basic tool to a "Major Project"-worthy premium application.
- **Features**: Interactive, semi-transparent overlays, Framer Motion animations, and Grad-CAM "Heatmap" displays so the farmer sees *why* the AI made its prediction.

## 🛠️ How to Verify and Run

### 1. Verify ML Results
You can verify the robust accuracy of the custom model by running the evaluation scripts provided:

```powershell
# Generate the official 38-class Confusion Matrix & Classification Report
python ml-training/evaluate_agrinet.py

# View a visual grid testing the model strictly on messy field data (PlantDoc)
python ml-training/test_field_data.py
```

### 2. Run the RAG Chatbot
Verify the RAG engine by setting up your environment and running the interface:

```powershell
# 1. Create a .env file at the root directory and add your Groq API key:
# GROQ_API_KEY=your_groq_api_key_here

# 2. Ingest the agricultural manuals (pre-built FAISS index is already in rag-engine/faiss_index)
python rag-engine/ingest.py

# 3. Start the query interface (can run in interactive loop or single-query mode)
python rag-engine/chat.py "How do I treat Apple Scab?"
```
