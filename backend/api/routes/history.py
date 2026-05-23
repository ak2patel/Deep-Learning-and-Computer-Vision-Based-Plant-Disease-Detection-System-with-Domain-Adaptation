"""
Prediction history routes — backed by PostgreSQL via Prisma.

All endpoints require JWT authentication. History is scoped to the
currently authenticated user (users can only see/modify their own data).

Endpoints:
    POST   /api/history/add           - Save a prediction to history
    GET    /api/history/              - Get user's prediction history
    DELETE /api/history/{prediction_id} - Delete a specific prediction
    DELETE /api/history/              - Clear all history for the user
    GET    /api/history/stats         - Get prediction statistics
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from utils.database import db
from utils.auth import get_current_user

router = APIRouter()


# ---------- Request / Response Schemas ----------

class AddPredictionRequest(BaseModel):
    plant_name: str
    disease_name: str
    confidence: float
    is_healthy: bool
    image_name: Optional[str] = None


class PredictionRecord(BaseModel):
    id: str
    plant_name: str
    disease_name: str
    confidence: float
    is_healthy: bool
    image_name: Optional[str] = None
    created_at: str


class HistoryResponse(BaseModel):
    predictions: List[PredictionRecord]
    total: int


# ---------- Endpoints ----------

@router.post("/add", status_code=status.HTTP_201_CREATED)
async def add_to_history(
    record: AddPredictionRequest,
    current_user=Depends(get_current_user),
):
    """
    Save a new prediction to the authenticated user's history.
    
    Requires: Authorization: Bearer <token>
    """
    try:
        prediction = await db.prediction.create(
            data={
                "userId": current_user.id,
                "plantName": record.plant_name,
                "diseaseName": record.disease_name,
                "confidence": record.confidence,
                "isHealthy": record.is_healthy,
                "imageName": record.image_name,
            }
        )
        
        return {
            "message": "Prediction saved to history",
            "id": prediction.id,
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save history: {str(e)}",
        )


@router.get("/", response_model=HistoryResponse)
async def get_history(
    limit: int = 50,
    current_user=Depends(get_current_user),
):
    """
    Get the authenticated user's prediction history (newest first).
    
    Requires: Authorization: Bearer <token>
    """
    try:
        predictions = await db.prediction.find_many(
            where={"userId": current_user.id},
            order={"createdAt": "desc"},
            take=limit,
        )
        
        records = [
            PredictionRecord(
                id=p.id,
                plant_name=p.plantName,
                disease_name=p.diseaseName,
                confidence=p.confidence,
                is_healthy=p.isHealthy,
                image_name=p.imageName,
                created_at=p.createdAt.isoformat(),
            )
            for p in predictions
        ]
        
        return HistoryResponse(
            predictions=records,
            total=len(records),
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load history: {str(e)}",
        )


@router.delete("/{prediction_id}")
async def delete_from_history(
    prediction_id: str,
    current_user=Depends(get_current_user),
):
    """
    Delete a specific prediction from history.
    Only the owner can delete their own predictions.
    
    Requires: Authorization: Bearer <token>
    """
    try:
        # Verify ownership
        prediction = await db.prediction.find_unique(where={"id": prediction_id})
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prediction not found",
            )
        if prediction.userId != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only delete your own predictions",
            )
        
        await db.prediction.delete(where={"id": prediction_id})
        return {"message": "Prediction deleted from history"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete: {str(e)}",
        )


@router.delete("/")
async def clear_history(current_user=Depends(get_current_user)):
    """
    Clear all prediction history for the authenticated user.
    
    Requires: Authorization: Bearer <token>
    """
    try:
        deleted = await db.prediction.delete_many(
            where={"userId": current_user.id}
        )
        return {
            "message": "History cleared",
            "deleted_count": deleted,
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear history: {str(e)}",
        )


@router.get("/stats")
async def get_statistics(current_user=Depends(get_current_user)):
    """
    Get prediction statistics for the authenticated user.
    
    Requires: Authorization: Bearer <token>
    """
    try:
        # Fetch all predictions for this user
        predictions = await db.prediction.find_many(
            where={"userId": current_user.id}
        )
        
        if not predictions:
            return {
                "total_predictions": 0,
                "healthy_count": 0,
                "diseased_count": 0,
                "most_common_disease": None,
                "average_confidence": 0,
            }
        
        healthy_count = sum(1 for p in predictions if p.isHealthy)
        diseased_count = len(predictions) - healthy_count
        
        # Most common disease
        diseases = [p.diseaseName for p in predictions if not p.isHealthy]
        most_common = max(set(diseases), key=diseases.count) if diseases else None
        
        # Average confidence
        avg_confidence = sum(p.confidence for p in predictions) / len(predictions)
        
        return {
            "total_predictions": len(predictions),
            "healthy_count": healthy_count,
            "diseased_count": diseased_count,
            "most_common_disease": most_common,
            "average_confidence": round(avg_confidence, 2),
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}",
        )
