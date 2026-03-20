"""
FastAPI application — the philosopher-king's gateway to the knowledge graph.

Provides REST endpoints for managing hypotheses, variables, causal DAGs,
evidence, and running causal inference queries. This API connects the
frontend (philosopher-king interface) to the backend graph store and
causal engine.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from causal.dag import CausalDAGEngine
from causal.pearl import backdoor_criterion, frontdoor_criterion, identify_effect
from graph.entities import (
    CausalDAG,
    Evidence,
    Hypothesis,
    HypothesisStatus,
    Variable,
)
from graph.store import KnowledgeGraphStore

from api.knowledge_routes import router as knowledge_router, set_store as set_knowledge_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Republic of AI Agents — Knowledge Graph API",
    description=(
        "REST API for the knowledge graph backend. Provides endpoints for "
        "hypothesis management, causal DAG operations, evidence tracking, "
        "Pearl's causal inference queries, and knowledge foundation management "
        "(thinkers, concepts, traditions, works, historical periods, domains)."
    ),
    version="0.2.0",
)

# In-memory store — single instance for the application lifecycle.
# Auto-load snapshot if available; otherwise start empty.
_SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "seeds" / "knowledge_base.json"

if _SNAPSHOT_PATH.exists():
    logger.info("Loading knowledge base snapshot from %s", _SNAPSHOT_PATH)
    store = KnowledgeGraphStore.from_json(_SNAPSHOT_PATH.read_text())
    _stats = store.stats()
    logger.info(
        "Loaded %d nodes, %d edges (%d thinkers, %d concepts, %d works)",
        _stats.get("graph_nodes", 0),
        _stats.get("graph_edges", 0),
        _stats.get("thinkers", 0),
        _stats.get("concepts", 0),
        _stats.get("works", 0),
    )
else:
    logger.info("No snapshot found — starting with empty store")
    store = KnowledgeGraphStore()

# Wire the knowledge routes to the shared store
set_knowledge_store(store)
app.include_router(knowledge_router)


def get_store() -> KnowledgeGraphStore:
    """Return the global knowledge graph store instance."""
    return store


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class HypothesisStatusUpdate(BaseModel):
    """Request body for updating a hypothesis's status."""

    status: HypothesisStatus = Field(
        ..., description="The new lifecycle status for the hypothesis."
    )


class CausalPathRequest(BaseModel):
    """Query parameters for causal path lookup."""

    source: str = Field(..., description="Source node ID.")
    target: str = Field(..., description="Target node ID.")


class BackdoorRequest(BaseModel):
    """Query parameters for backdoor criterion computation."""

    treatment: str = Field(..., description="Treatment variable node ID.")
    outcome: str = Field(..., description="Outcome variable node ID.")
    dag_id: str = Field(..., description="ID of the causal DAG to analyze.")


class IdentifyEffectResponse(BaseModel):
    """Response for effect identifiability query."""

    identifiable: bool
    method: Optional[str] = None
    adjustment_sets: list[list[str]] = Field(default_factory=list)


class NeighborsResponse(BaseModel):
    """Response for graph neighbors query."""

    node_id: str
    depth: int
    neighbors: dict[str, list[str]]


class StatsResponse(BaseModel):
    """Response for knowledge graph statistics."""

    hypotheses: int
    variables: int
    causal_dags: int
    evidence: int
    experiments: int
    thinkers: int = 0
    concepts: int = 0
    traditions: int = 0
    works: int = 0
    historical_periods: int = 0
    domains: int = 0
    graph_nodes: int
    graph_edges: int
    mutations: int


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Dictionary with status "ok" if the service is running.
    """
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse, tags=["system"])
async def get_stats() -> StatsResponse:
    """Return summary statistics about the knowledge graph.

    Returns:
        Counts of each entity type and graph metrics.
    """
    return StatsResponse(**store.stats())


# ---------------------------------------------------------------------------
# Hypothesis endpoints
# ---------------------------------------------------------------------------


@app.post("/hypotheses", response_model=Hypothesis, tags=["hypotheses"])
async def create_hypothesis(hypothesis: Hypothesis) -> Hypothesis:
    """Register a new hypothesis in the knowledge graph.

    The hypothesis must include at least a title, author, and description.
    Predictions with falsification criteria are strongly encouraged per
    Popperian methodology.

    Args:
        hypothesis: The Hypothesis to register.

    Returns:
        The registered Hypothesis with generated ID and timestamps.
    """
    try:
        return store.add_hypothesis(hypothesis)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/hypotheses", response_model=list[Hypothesis], tags=["hypotheses"])
async def list_hypotheses(
    status: Optional[HypothesisStatus] = Query(
        default=None, description="Filter by hypothesis status."
    ),
) -> list[Hypothesis]:
    """List all hypotheses, optionally filtered by status.

    Args:
        status: If provided, only return hypotheses with this lifecycle status.

    Returns:
        List of matching Hypothesis objects.
    """
    return store.list_hypotheses(status=status)


@app.get("/hypotheses/{hypothesis_id}", response_model=Hypothesis, tags=["hypotheses"])
async def get_hypothesis(hypothesis_id: str) -> Hypothesis:
    """Retrieve a specific hypothesis by ID.

    Args:
        hypothesis_id: The unique identifier of the hypothesis.

    Returns:
        The requested Hypothesis.

    Raises:
        HTTPException(404): If the hypothesis is not found.
    """
    h = store.get_hypothesis(hypothesis_id)
    if h is None:
        raise HTTPException(status_code=404, detail="Hypothesis not found.")
    return h


@app.patch(
    "/hypotheses/{hypothesis_id}/status",
    response_model=Hypothesis,
    tags=["hypotheses"],
)
async def update_hypothesis_status(
    hypothesis_id: str,
    body: HypothesisStatusUpdate,
) -> Hypothesis:
    """Update the lifecycle status of a hypothesis.

    This is the primary mechanism for moving hypotheses through the
    Kuhnian lifecycle: proposed -> testing -> validated/falsified -> paradigm.

    Args:
        hypothesis_id: The ID of the hypothesis to update.
        body: Request body containing the new status.

    Returns:
        The updated Hypothesis.

    Raises:
        HTTPException(404): If the hypothesis is not found.
    """
    try:
        return store.update_hypothesis_status(hypothesis_id, body.status)
    except KeyError:
        raise HTTPException(status_code=404, detail="Hypothesis not found.")


# ---------------------------------------------------------------------------
# Variable endpoints
# ---------------------------------------------------------------------------


@app.post("/variables", response_model=Variable, tags=["variables"])
async def create_variable(variable: Variable) -> Variable:
    """Register a new variable in the knowledge graph.

    Variables are the atomic units of causal models — they represent
    measurable or latent quantities that merchant agents collect data about.

    Args:
        variable: The Variable to register.

    Returns:
        The registered Variable with generated ID.
    """
    try:
        return store.add_variable(variable)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/variables", response_model=list[Variable], tags=["variables"])
async def list_variables() -> list[Variable]:
    """List all registered variables.

    Returns:
        List of all Variable objects in the knowledge graph.
    """
    return list(store.variables.values())


@app.get("/variables/{variable_id}", response_model=Variable, tags=["variables"])
async def get_variable(variable_id: str) -> Variable:
    """Retrieve a specific variable by ID.

    Args:
        variable_id: The unique identifier of the variable.

    Returns:
        The requested Variable.

    Raises:
        HTTPException(404): If the variable is not found.
    """
    v = store.get_variable(variable_id)
    if v is None:
        raise HTTPException(status_code=404, detail="Variable not found.")
    return v


# ---------------------------------------------------------------------------
# Causal DAG endpoints
# ---------------------------------------------------------------------------


@app.post("/causal-dags", response_model=CausalDAG, tags=["causal-dags"])
async def create_causal_dag(dag: CausalDAG) -> CausalDAG:
    """Register a new causal DAG in the knowledge graph.

    The DAG's nodes and edges are integrated into the underlying graph
    store, linking variables and causal relationships into the unified
    knowledge structure.

    Args:
        dag: The CausalDAG to register.

    Returns:
        The registered CausalDAG with generated ID and timestamps.
    """
    try:
        return store.add_causal_dag(dag)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/causal-dags", response_model=list[CausalDAG], tags=["causal-dags"])
async def list_causal_dags() -> list[CausalDAG]:
    """List all registered causal DAGs.

    Returns:
        List of all CausalDAG objects in the knowledge graph.
    """
    return list(store.causal_dags.values())


@app.get(
    "/causal-dags/{dag_id}", response_model=CausalDAG, tags=["causal-dags"]
)
async def get_causal_dag(dag_id: str) -> CausalDAG:
    """Retrieve a specific causal DAG by ID.

    Args:
        dag_id: The unique identifier of the DAG.

    Returns:
        The requested CausalDAG.

    Raises:
        HTTPException(404): If the DAG is not found.
    """
    d = store.get_causal_dag(dag_id)
    if d is None:
        raise HTTPException(status_code=404, detail="Causal DAG not found.")
    return d


# ---------------------------------------------------------------------------
# Evidence endpoints
# ---------------------------------------------------------------------------


@app.post("/evidence", response_model=Evidence, tags=["evidence"])
async def create_evidence(evidence_item: Evidence) -> Evidence:
    """Register new evidence linked to a hypothesis.

    Evidence is the empirical bedrock of the knowledge graph — it connects
    theoretical hypotheses to real-world observations gathered by merchant
    agents.

    Args:
        evidence_item: The Evidence to register.

    Returns:
        The registered Evidence with generated ID and timestamp.

    Raises:
        HTTPException(404): If the referenced hypothesis does not exist.
        HTTPException(409): If evidence with the same ID already exists.
    """
    try:
        return store.add_evidence(evidence_item)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get(
    "/evidence/by-hypothesis/{hypothesis_id}",
    response_model=list[Evidence],
    tags=["evidence"],
)
async def get_evidence_for_hypothesis(hypothesis_id: str) -> list[Evidence]:
    """Retrieve all evidence linked to a specific hypothesis.

    Args:
        hypothesis_id: The ID of the hypothesis.

    Returns:
        List of Evidence objects, both supporting and contradicting.
    """
    return store.get_evidence_for_hypothesis(hypothesis_id)


# ---------------------------------------------------------------------------
# Graph query endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/graph/neighbors/{node_id}",
    response_model=NeighborsResponse,
    tags=["graph"],
)
async def get_neighbors(
    node_id: str,
    depth: int = Query(default=1, ge=1, le=5, description="Traversal depth."),
) -> NeighborsResponse:
    """Find all neighbors of a node up to a given depth.

    Performs a BFS traversal from the specified node in the undirected
    version of the graph, returning neighbors grouped by distance level.

    Args:
        node_id: The starting node ID.
        depth: Maximum traversal depth (1-5).

    Returns:
        Neighbors grouped by distance from the starting node.

    Raises:
        HTTPException(404): If the node does not exist.
    """
    try:
        neighbors = store.query_neighbors(node_id, depth=depth)
        return NeighborsResponse(
            node_id=node_id,
            depth=depth,
            neighbors=neighbors,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")


@app.get("/graph/causal-path", response_model=list[list[str]], tags=["graph"])
async def get_causal_path(
    source: str = Query(..., description="Source node ID."),
    target: str = Query(..., description="Target node ID."),
) -> list[list[str]]:
    """Find all directed causal paths between two nodes.

    Args:
        source: Starting node ID.
        target: Ending node ID.

    Returns:
        List of paths, each a list of node IDs.

    Raises:
        HTTPException(404): If source or target node does not exist.
    """
    try:
        return store.get_causal_path(source, target)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Causal inference endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/causal/backdoor",
    response_model=list[list[str]],
    tags=["causal"],
)
async def get_backdoor_adjustment(
    dag_id: str = Query(..., description="ID of the causal DAG to analyze."),
    treatment: str = Query(..., description="Treatment variable node ID."),
    outcome: str = Query(..., description="Outcome variable node ID."),
) -> list[list[str]]:
    """Find valid backdoor adjustment sets for a causal effect.

    Uses Pearl's backdoor criterion to identify sets of variables that,
    when conditioned on, block all confounding paths between treatment
    and outcome.

    Args:
        dag_id: The causal DAG to analyze.
        treatment: The treatment variable.
        outcome: The outcome variable.

    Returns:
        List of adjustment sets (each set as a list of variable IDs).

    Raises:
        HTTPException(404): If DAG, treatment, or outcome not found.
    """
    dag = store.get_causal_dag(dag_id)
    if dag is None:
        raise HTTPException(status_code=404, detail="Causal DAG not found.")

    try:
        engine = CausalDAGEngine.from_schema(dag)
        sets = backdoor_criterion(engine, treatment, outcome)
        return [sorted(s) for s in sets]
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get(
    "/causal/frontdoor",
    response_model=list[list[str]],
    tags=["causal"],
)
async def get_frontdoor_sets(
    dag_id: str = Query(..., description="ID of the causal DAG to analyze."),
    treatment: str = Query(..., description="Treatment variable node ID."),
    outcome: str = Query(..., description="Outcome variable node ID."),
) -> list[list[str]]:
    """Find variable sets satisfying the frontdoor criterion.

    The frontdoor criterion enables causal effect identification through
    mediators, even when unmeasured confounders exist between treatment
    and outcome.

    Args:
        dag_id: The causal DAG to analyze.
        treatment: The treatment variable.
        outcome: The outcome variable.

    Returns:
        List of mediator sets (each set as a list of variable IDs).

    Raises:
        HTTPException(404): If DAG, treatment, or outcome not found.
    """
    dag = store.get_causal_dag(dag_id)
    if dag is None:
        raise HTTPException(status_code=404, detail="Causal DAG not found.")

    try:
        engine = CausalDAGEngine.from_schema(dag)
        sets = frontdoor_criterion(engine, treatment, outcome)
        return [sorted(s) for s in sets]
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get(
    "/causal/identify",
    response_model=IdentifyEffectResponse,
    tags=["causal"],
)
async def identify_causal_effect(
    dag_id: str = Query(..., description="ID of the causal DAG to analyze."),
    treatment: str = Query(..., description="Treatment variable node ID."),
    outcome: str = Query(..., description="Outcome variable node ID."),
) -> IdentifyEffectResponse:
    """Check whether a causal effect is identifiable from observational data.

    Attempts identification via backdoor and frontdoor criteria, reporting
    which method succeeds and what adjustment sets are available.

    Args:
        dag_id: The causal DAG to analyze.
        treatment: The treatment variable.
        outcome: The outcome variable.

    Returns:
        Identifiability result with method and adjustment sets.

    Raises:
        HTTPException(404): If DAG, treatment, or outcome not found.
    """
    dag = store.get_causal_dag(dag_id)
    if dag is None:
        raise HTTPException(status_code=404, detail="Causal DAG not found.")

    try:
        engine = CausalDAGEngine.from_schema(dag)
        result = identify_effect(engine, treatment, outcome)
        return IdentifyEffectResponse(
            identifiable=result["identifiable"],
            method=result["method"],
            adjustment_sets=[sorted(s) for s in result["adjustment_sets"]],
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
