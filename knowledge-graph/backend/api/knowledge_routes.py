"""
Knowledge foundation API routes.

Provides REST endpoints for managing the intellectual foundation layer:
thinkers, concepts, traditions, works, historical periods, and domains.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from graph.knowledge_entities import (
    Concept,
    Domain,
    Era,
    HistoricalPeriod,
    Thinker,
    Tradition,
    Work,
)
from graph.store import KnowledgeGraphStore

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

# Will be set by main.py at startup
_store: KnowledgeGraphStore | None = None


def set_store(store: KnowledgeGraphStore) -> None:
    global _store
    _store = store


def get_store() -> KnowledgeGraphStore:
    if _store is None:
        raise RuntimeError("Store not initialized")
    return _store


# ---------------------------------------------------------------------------
# Thinker endpoints
# ---------------------------------------------------------------------------


@router.post("/thinkers", response_model=Thinker)
async def create_thinker(thinker: Thinker) -> Thinker:
    try:
        return get_store().add_thinker(thinker)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/thinkers", response_model=list[Thinker])
async def list_thinkers(
    era: Optional[Era] = Query(default=None),
    tradition_id: Optional[str] = Query(default=None),
    domain_id: Optional[str] = Query(default=None),
    tier: Optional[int] = Query(default=None, ge=1, le=3),
) -> list[Thinker]:
    return get_store().list_thinkers(
        era=era, tradition_id=tradition_id, domain_id=domain_id, tier=tier
    )


@router.get("/thinkers/{thinker_id}", response_model=Thinker)
async def get_thinker(thinker_id: str) -> Thinker:
    t = get_store().get_thinker(thinker_id)
    if t is None:
        raise HTTPException(status_code=404, detail="Thinker not found.")
    return t


# ---------------------------------------------------------------------------
# Concept endpoints
# ---------------------------------------------------------------------------


@router.post("/concepts", response_model=Concept)
async def create_concept(concept: Concept) -> Concept:
    try:
        return get_store().add_concept(concept)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/concepts", response_model=list[Concept])
async def list_concepts(
    domain_id: Optional[str] = Query(default=None),
    tradition_id: Optional[str] = Query(default=None),
) -> list[Concept]:
    return get_store().list_concepts(domain_id=domain_id, tradition_id=tradition_id)


@router.get("/concepts/{concept_id}", response_model=Concept)
async def get_concept(concept_id: str) -> Concept:
    c = get_store().get_concept(concept_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Concept not found.")
    return c


# ---------------------------------------------------------------------------
# Tradition endpoints
# ---------------------------------------------------------------------------


@router.post("/traditions", response_model=Tradition)
async def create_tradition(tradition: Tradition) -> Tradition:
    try:
        return get_store().add_tradition(tradition)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/traditions", response_model=list[Tradition])
async def list_traditions() -> list[Tradition]:
    return get_store().list_traditions()


@router.get("/traditions/{tradition_id}", response_model=Tradition)
async def get_tradition(tradition_id: str) -> Tradition:
    t = get_store().get_tradition(tradition_id)
    if t is None:
        raise HTTPException(status_code=404, detail="Tradition not found.")
    return t


# ---------------------------------------------------------------------------
# Work endpoints
# ---------------------------------------------------------------------------


@router.post("/works", response_model=Work)
async def create_work(work: Work) -> Work:
    try:
        return get_store().add_work(work)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/works", response_model=list[Work])
async def list_works(
    author_id: Optional[str] = Query(default=None),
    tradition_id: Optional[str] = Query(default=None),
) -> list[Work]:
    return get_store().list_works(author_id=author_id, tradition_id=tradition_id)


@router.get("/works/{work_id}", response_model=Work)
async def get_work(work_id: str) -> Work:
    w = get_store().get_work(work_id)
    if w is None:
        raise HTTPException(status_code=404, detail="Work not found.")
    return w


# ---------------------------------------------------------------------------
# Historical period endpoints
# ---------------------------------------------------------------------------


@router.post("/historical-periods", response_model=HistoricalPeriod)
async def create_historical_period(period: HistoricalPeriod) -> HistoricalPeriod:
    try:
        return get_store().add_historical_period(period)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/historical-periods", response_model=list[HistoricalPeriod])
async def list_historical_periods() -> list[HistoricalPeriod]:
    return get_store().list_historical_periods()


@router.get("/historical-periods/{period_id}", response_model=HistoricalPeriod)
async def get_historical_period(period_id: str) -> HistoricalPeriod:
    hp = get_store().get_historical_period(period_id)
    if hp is None:
        raise HTTPException(status_code=404, detail="Historical period not found.")
    return hp


# ---------------------------------------------------------------------------
# Domain endpoints
# ---------------------------------------------------------------------------


@router.post("/domains", response_model=Domain)
async def create_domain(domain: Domain) -> Domain:
    try:
        return get_store().add_domain(domain)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/domains", response_model=list[Domain])
async def list_domains() -> list[Domain]:
    return get_store().list_domains()


@router.get("/domains/{domain_id}", response_model=Domain)
async def get_domain(domain_id: str) -> Domain:
    d = get_store().get_domain(domain_id)
    if d is None:
        raise HTTPException(status_code=404, detail="Domain not found.")
    return d


# ---------------------------------------------------------------------------
# Special query endpoints
# ---------------------------------------------------------------------------


@router.get("/lineage/{thinker_id}")
async def get_lineage(thinker_id: str) -> dict:
    try:
        return get_store().get_intellectual_lineage(thinker_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Thinker not found.")


@router.get("/dialectic/{concept_id}")
async def get_dialectic(concept_id: str) -> dict:
    try:
        return get_store().get_concept_dialectic(concept_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Concept not found.")


@router.get("/by-chapter/{chapter}")
async def get_by_chapter(chapter: int) -> dict:
    return get_store().search_by_manuscript_chapter(chapter)


@router.post("/resolve-edges")
async def resolve_edges() -> dict:
    count = get_store().resolve_deferred_edges()
    return {"new_edges": count}


@router.get("/stats")
async def knowledge_stats() -> dict:
    s = get_store()
    return {
        "thinkers": len(s.thinkers),
        "concepts": len(s.concepts),
        "traditions": len(s.traditions),
        "works": len(s.works),
        "historical_periods": len(s.historical_periods),
        "domains": len(s.domains),
    }


@router.post("/snapshot")
async def save_snapshot() -> dict:
    store = get_store()
    snapshot_path = Path(__file__).parent.parent / "seeds" / "knowledge_base.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(store.to_json())
    return {"path": str(snapshot_path), "stats": store.stats()}
