# ✨ Frontend & UI Architect Guide - Nawazish (`2201031CS`)

Welcome to the UI/UX phase! You are responsible for the entire user experience. No matter how good the ML model is, the presentation is what reviewers see first. Your job is to make it look premium.

## Your Domain
Everything you do should be entirely inside the `/frontend` folder. **Do not touch the `ml-training` or `backend` folders.**

## Your Primary Tasks
1. **Web App Framework:** Initialize a **Next.js** application inside the `/frontend` directory. 
2. **Premium Design:** Implement the "Astro-Agricultural" theme. Use **Glassmorphism** (semi-transparent, blurred backgrounds), modern fonts (Inter/Outfit), and **Framer Motion** for smooth page transitions.
3. **Core Interface:** Build a stunning Landing Page and an interactive **Prediction Dashboard**. 
4. **API Connection:** Create an image upload component that sends the picture to Ankit's FastAPI `POST /predict` endpoint, and elegantly displays the result to the user. Make sure to render the `gradcam_image` (a base64 string returned by the API) alongside the original uploaded image to show the Explainable AI heatmap.
5. **PDF Reporting:** Build a feature for the user to download their disease diagnosis as a professional PDF report.

## Helpful AI Prompt for you to use:
*When you are ready to start coding, paste this exact prompt into the AI assistant:*

> "Pretend to be the Frontend Architect for this Major Project. Please read my guide at `Team_Guides/Nawazish_Frontend_Guide.md` and the `UI_UX_Design_Guide.md` in the root folder. My first task is to set up a Next.js application in the `/frontend` folder and create a Glassmorphism landing page and prediction dashboard. Walk me through the setup commands and write the components. Ensure the prediction dashboard uses the `gradcam_image` base64 string from the API response to display the Grad-CAM heatmap."
