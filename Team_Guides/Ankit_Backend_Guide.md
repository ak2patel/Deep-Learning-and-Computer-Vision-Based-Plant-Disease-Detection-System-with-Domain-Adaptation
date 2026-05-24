# ⚙️ Backend Developer Guide - Ankit (`2201116CS`)

Welcome to the backend phase of the AI Powered Plant Disease Detection System! Your role is the backbone of the application. You will connect Aman's AI model to the internet and store user data securely.

## Your Domain
Everything you do should be entirely inside the `/backend` folder. **Do not touch the `ml-training` or `frontend` folders.**

## Your Primary Tasks
1. **API Server:** Create a `FastAPI` application.
2. **Model Serving:** Maintain the existing `POST /predict` endpoint. It accepts an image upload, passes the image through Aman's `agrinet_mixed_v1.pth` PyTorch model, and returns the predicted disease name along with a `gradcam_image` (a base64 encoded heatmap). Do not overwrite the existing Grad-CAM generation logic!
3. **Database Integration:** Set up **PostgreSQL**. Use **Prisma ORM** (via `prisma-client-python`) instead of SQLAlchemy to create a `User` model (for login) and a `Prediction` model (to save the history of diseases a user scans). Prisma is modern and much cleaner for Major Projects.
4. **Auth Flow:** Add JWT-based login/signup endpoints.

## Helpful AI Prompt for you to use:
*When you are ready to start coding, paste this exact prompt into the AI assistant:*

> "Pretend to be the Backend Engineer for this Major Project. Please read my guide at `Team_Guides/Ankit_Backend_Guide.md`. My first task is to set up the FastAPI boilerplate inside a new `/backend` folder. Please write `backend/main.py` with a basic health check endpoint, initialize Prisma ORM for PostgreSQL, and give instructions on how to start the Uvicorn server. Make sure to respect the existing `/predict` endpoint and Grad-CAM logic."
