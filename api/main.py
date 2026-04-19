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
    description='Tiered recommendation engine with reranker and all-model comparison',
    version='2.0.0',
    lifespan=lifespan,
)


# ---- Request models ----

class RecommendRequest(BaseModel):
    user_id: Optional[int] = Field(None, description='Known user ID (0-indexed)')
    history: list[int] = Field(..., description='Item IDs in chronological order')
    n: int = Field(10, ge=1, le=100)


class RecommendAllRequest(BaseModel):
    user_id: Optional[int] = Field(None)
    history: list[int] = Field(...)
    n: int = Field(10, ge=1, le=50)


class SimilarItemsRequest(BaseModel):
    item_id: int = Field(...)
    n: int = Field(10, ge=1, le=100)


class ItemSimilarityRequest(BaseModel):
    item_id: int = Field(...)
    n: int = Field(10, ge=1, le=20)


class NewItemRequest(BaseModel):
    title: str = Field(..., description='Movie title')
    genres: list[str] = Field(..., description='List of genres from the 18 ML-1M genres')
    n_similar: int = Field(10, ge=1, le=20)
    n_users: int = Field(20, ge=5, le=100)


class TwoTowerDemoRequest(BaseModel):
    history: list[int] = Field(...)
    gender_enc: int = Field(..., ge=0, le=1, description='0=Female 1=Male')
    age_enc: int = Field(..., ge=0, le=6, description='0=<18 .. 6=56+')
    n: int = Field(10, ge=1, le=50)


class NewUserRequest(BaseModel):
    gender_enc: int = Field(..., ge=0, le=1)
    age_enc: int = Field(..., ge=0, le=6)
    n: int = Field(10, ge=1, le=50)


class HomepageRequest(BaseModel):
    history: list[int] = Field(...)
    gender_enc: Optional[int] = Field(None, ge=0, le=1)
    age_enc: Optional[int] = Field(None, ge=0, le=6)
    n: int = Field(10, ge=1, le=20)


# ---- Endpoints ----

@app.get('/health')
def health():
    return {
        'status': 'ok',
        'models': {
            'tier1': 'popularity+reranker',
            'tier2': 'cf+svd+reranker',
            'tier3': 'sasrec',
            'cold_item': 'two_tower',
        },
        'capabilities': [
            'recommend', 'recommend_all', 'similar_items', 'item_similarity',
            'new_item_cold_start', 'two_tower_with_demo', 'new_user_cold_start',
            'homepage',
        ],
    }


@app.post('/recommend')
def recommend(req: RecommendRequest):
    if engine is None:
        raise HTTPException(503, 'Engine not ready')
    return engine.recommend(
        user_id=req.user_id, history=req.history, N=req.n,
    )


@app.post('/recommend_all')
def recommend_all(req: RecommendAllRequest):
    """Run all 5 models + reranker. Used by Tab 1 of the UI."""
    if engine is None:
        raise HTTPException(503, 'Engine not ready')
    return engine.recommend_all_models(
        user_id=req.user_id, history=req.history, N=req.n,
    )


@app.post('/similar_items')
def similar_items(req: SimilarItemsRequest):
    if engine is None:
        raise HTTPException(503, 'Engine not ready')
    return engine.recommend_cold_item(item_id=req.item_id, N=req.n)


@app.post('/item_similarity')
def item_similarity(req: ItemSimilarityRequest):
    """Top-N similar items from CF, SVD, Two-Tower. Used by Tab 4 of the UI."""
    if engine is None:
        raise HTTPException(503, 'Engine not ready')
    return engine.item_similarity(item_id=req.item_id, N=req.n)


@app.post('/new_item_cold_start')
def new_item_cold_start(req: NewItemRequest):
    """Encode a new item and find similar items + likely users + demographic bias."""
    if engine is None:
        raise HTTPException(503, 'Engine not ready')
    return engine.new_item_cold_start(
        title=req.title, genres=req.genres,
        n_similar=req.n_similar, n_users=req.n_users,
    )


@app.post('/two_tower_with_demo')
def two_tower_with_demo(req: TwoTowerDemoRequest):
    """Two-Tower with history + explicit demographics. Used by Tab 1 demographic filter."""
    if engine is None:
        raise HTTPException(503, 'Engine not ready')
    return {
        'items': engine.two_tower_with_demo(
            history=req.history, gender_enc=req.gender_enc,
            age_enc=req.age_enc, N=req.n,
        ),
    }


@app.post('/new_user_cold_start')
def new_user_cold_start(req: NewUserRequest):
    """Cold start for a new user with demographics only. Returns trending + Two-Tower."""
    if engine is None:
        raise HTTPException(503, 'Engine not ready')
    return engine.new_user_cold_start(
        gender_enc=req.gender_enc, age_enc=req.age_enc, N=req.n,
    )


@app.post('/homepage')
def homepage(req: HomepageRequest):
    """Netflix-style homepage rows. Routes by history length."""
    if engine is None:
        raise HTTPException(503, 'Engine not ready')

    n_history = len(req.history)
    new_arrivals = engine.get_simulated_new_arrivals(N=10)

    if n_history > 5:
        sasrec_items, _, _ = engine._tier3(req.history, req.n, exclude_seen=True)
        trending = engine.popular_items[:req.n]
        genre_rows = engine.get_genre_rows_for_user(req.history, n_candidates=200, top_n=5)
        genres = list(genre_rows.keys())
        return {
            'user_type': 'warm',
            'n_history': n_history,
            'rows': [
                {'label': 'Based on your watch history', 'model': 'sasrec',
                 'items': sasrec_items},
                {'label': 'Trending', 'model': 'popularity',
                 'items': trending},
                {'label': f'Because you watched {genres[0]}' if len(genres) > 0 else 'Top picks',
                 'model': 'two_tower_filtered',
                 'genre': genres[0] if genres else None,
                 'items': genre_rows.get(genres[0], []) if genres else []},
                {'label': f'Because you watched {genres[1]}' if len(genres) > 1 else 'Top picks',
                 'model': 'two_tower_filtered',
                 'genre': genres[1] if len(genres) > 1 else None,
                 'items': genre_rows.get(genres[1], []) if len(genres) > 1 else []},
                {'label': 'New arrivals', 'model': 'simulated',
                 'items': new_arrivals},
            ],
        }
    else:
        gender_enc = req.gender_enc if req.gender_enc is not None else 0
        age_enc = req.age_enc if req.age_enc is not None else 2
        trending = engine.popular_items[:req.n]
        cold_rows = engine.get_cold_genre_rows(gender_enc, age_enc, top_n=5)
        return {
            'user_type': 'cold',
            'n_history': n_history,
            'rows': [
                {'label': 'Trending', 'model': 'popularity',
                 'items': trending},
                {'label': f"Popular in {cold_rows['age_gender_1']['genre']}",
                 'model': 'demographic_genre',
                 'genre': cold_rows['age_gender_1']['genre'],
                 'items': cold_rows['age_gender_1']['items']},
                {'label': f"Popular in {cold_rows['age_gender_2']['genre']}",
                 'model': 'demographic_genre',
                 'genre': cold_rows['age_gender_2']['genre'],
                 'items': cold_rows['age_gender_2']['items']},
                {'label': f"Popular in {cold_rows['gender_only']['genre']}",
                 'model': 'demographic_genre',
                 'genre': cold_rows['gender_only']['genre'],
                 'items': cold_rows['gender_only']['items']},
                {'label': 'New arrivals', 'model': 'simulated',
                 'items': new_arrivals},
            ],
        }


if __name__ == '__main__':
    uvicorn.run('api.main:app', host='0.0.0.0', port=8000, reload=False)
