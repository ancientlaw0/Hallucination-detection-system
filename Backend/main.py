
import json
import os
from pathlib import Path
import uuid
from fastapi import UploadFile, File, Form
from Backend.retrieval import document_store, UPLOAD_DIR
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Locate the project and model directory.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(
    os.getenv(
        "MODEL_DIR",
        str(BASE_DIR / "artifacts" / "fine_tuned_hallucination_verifier"),
    )
)

if not MODEL_DIR.exists():
    raise RuntimeError(
        f"Model directory not found: {MODEL_DIR}. "
        "Place your exported Colab model in the artifacts folder."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load once when the backend starts.
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.to(device)
model.eval()

# Use the threshold selected using validation data in your notebook.
threshold = 0.5
metrics_file = MODEL_DIR / "evaluation_metrics.json"

if metrics_file.exists():
    with open(metrics_file, encoding="utf-8") as file:
        metrics = json.load(file)

    threshold = float(
        metrics.get("decision_threshold_selected_on_validation", 0.5)
    )

app = FastAPI(
    title="Hallucination Detection API",
    description="Response-level hallucination detection using fine-tuned DeBERTa.",
    version="1.0.0",
)


class VerifyRequest(BaseModel):
    evidence: str = Field(min_length=1, max_length=100000)
    response: str = Field(min_length=1, max_length=50000)


@app.get("/")
def root():
    return {"message": "Hallucination Detection API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(device),
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "Fine-tuned DeBERTa",
        "task": "response-level hallucination detection",
        "threshold": threshold,
        "device": str(device),
    }


@app.post("/verify")
def verify(request: VerifyRequest):
    try:
        inputs = tokenizer(
            request.evidence,
            request.response,
            truncation="longest_first",
            max_length=384,
            return_tensors="pt",
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        probability = float(probabilities[1].item())
        is_hallucination = probability >= threshold

        return {
            "prediction": (
                "potential_hallucination"
                if is_hallucination
                else "no_annotated_hallucination_detected"
            ),
            "hallucination_probability": round(probability, 4),
            "threshold": round(threshold, 4),
            "warning": (
                "This is a response-level prediction, not proof of factual "
                "incorrectness. Exact faulty claims are not identified."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Verification failed. Check backend logs.",
        ) from exc


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "document.pdf").name

    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported currently.",
        )

    document_id = str(uuid.uuid4())
    destination = UPLOAD_DIR / f"{document_id}.pdf"

    try:
        content = await file.read()

        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        # Simple upload limit for the local prototype.
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="PDF exceeds 20 MB.")

        destination.write_bytes(content)

        result = document_store.add_pdf(destination, document_id)
        result["original_filename"] = filename
        return result

    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Could not process this PDF. Check that it is a valid, text-based PDF.",
        ) from exc


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=10)


@app.post("/retrieve")
def retrieve(request: RetrieveRequest):
    results = document_store.search(request.query, request.top_k)

    return {
        "query": request.query,
        "results": results,
        "count": len(results),
    }
