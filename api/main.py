from contextlib import asynccontextmanager
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from config.config_loader import get_config
from inference.engine import TieredRecommendationEngine
from utils.logger import get_logger


logger = get_logger('api')
engine: TieredRecommendationEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info('Loading inference engine...')
    cfg = get_config()
    engine = TieredRecommendationEngine(cfg)
    logger.info('Engine ready.')
    yield
    logger.info('Shutting down.')


app = FastAPI(
    title='Serko RecSys API',
    description='Tiered recommendation engine: Popularity / SVD / SASRec',
    version='1.0.0',
    lifespan=lifespan,
)


class RecommendRequest(BaseModel):
    user_id: Optional[int] = Field(None, description='Known user ID (0-indexed)')
    history: list[int] = Field(..., description='Item IDs in chronological order')
    n: int = Field(10, ge=1, le=100, description='Number of recommendations')


class SimilarItemsRequest(BaseModel):
    item_id: int = Field(..., description='Item ID to find similar items for')
    n: int = Field(10, ge=1, le=100)


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'models': {
            'tier1': 'popularity',
            'tier2': 'svd_foldin',
            'tier3': 'sasrec',
            'cold_item': 'two_tower',
        },
    }


@app.post('/recommend')
def recommend(req: RecommendRequest):
    if engine is None:
        raise HTTPException(503, 'Engine not ready')
    return engine.recommend(
        user_id=req.user_id, history=req.history, N=req.n,
    )


@app.post('/similar_items')
def similar_items(req: SimilarItemsRequest):
    if engine is None:
        raise HTTPException(503, 'Engine not ready')
    return engine.recommend_cold_item(item_id=req.item_id, N=req.n)


if __name__ == '__main__':
    uvicorn.run('api.main:app', host='0.0.0.0', port=8000, reload=False)
