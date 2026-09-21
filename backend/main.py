from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid

# Import our custom AI modules
from backend.models.ai_model import GradCAMDenseNet
from backend.models.progression import ProgressionAnalyzer
from backend.models.retrieval import CaseRetriever
from backend.models.report_generator import RadiologyReportGenerator

app = FastAPI(title="PulmoVision API Gateway")

# Enable CORS so the web frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Modules
print("Loading PulmoVision AI Engine...")
detector = GradCAMDenseNet()
progression_analyzer = ProgressionAnalyzer()
retriever = CaseRetriever()
report_gen = RadiologyReportGenerator()

# Directory configuration
UPLOAD_DIR = os.path.join(os.getcwd(), "backend", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve uploaded images statically
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/")
def root():
    return {"status": "PulmoVision Backend API running successfully"}


@app.post("/api/analyze")
async def analyze_xray(
    file: UploadFile = File(...),
    prior_file: UploadFile = File(None),
    patient_age: int = Form(40),
    patient_gender: str = Form("Unknown"),
    symptoms: str = Form("Respiratory symptoms")
):
    try:
        # Save current uploaded file
        req_id = str(uuid.uuid4())[:8]
        current_ext = file.filename.split(".")[-1]
        current_path = os.path.join(UPLOAD_DIR, f"current_{req_id}.{current_ext}")
        current_gradcam_path = os.path.join(UPLOAD_DIR, f"gradcam_current_{req_id}.jpg")

        with open(current_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 1: AI Prediction & Grad-CAM
        ai_res = detector.predict_and_explain(current_path, current_gradcam_path)

        # Step 2: Progression Analysis (if prior image provided)
        progression_status = "No prior radiograph provided for baseline comparison."
        if prior_file:
            prior_ext = prior_file.filename.split(".")[-1]
            prior_path = os.path.join(UPLOAD_DIR, f"prior_{req_id}.{prior_ext}")
            with open(prior_path, "wb") as buffer:
                shutil.copyfileobj(prior_file.file, buffer)

            prog_res = progression_analyzer.analyze_progression(prior_path, current_path, UPLOAD_DIR)
            progression_status = prog_res["progression_status"]

        # Step 3: RAG Case Retrieval
        query = f"Patient presenting with {symptoms}. Suspected {ai_res['predicted_disease']}"
        retrieved_cases = retriever.retrieve_similar_cases(query_text=query, top_k=2)

        # Step 4: LLM Report Generation
        patient_info = {"age": patient_age, "gender": patient_gender, "symptoms": symptoms}
        generated_report = report_gen.generate_report(
            patient_info=patient_info,
            current_findings=ai_res,
            progression_status=progression_status,
            retrieved_cases=retrieved_cases
        )

        return {
            "status": "success",
            "findings": {
                "predicted_disease": ai_res["predicted_disease"],
                "confidence": ai_res["confidence"],
                "gradcam_url": f"/uploads/gradcam_current_{req_id}.jpg",
                "original_url": f"/uploads/current_{req_id}.{current_ext}"
            },
            "progression_status": progression_status,
            "retrieved_cases": retrieved_cases,
            "report": generated_report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)