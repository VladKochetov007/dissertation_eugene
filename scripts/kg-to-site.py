#!/usr/bin/env python3
"""
Convert knowledge_base.json from the knowledge graph backend
into graph-data.ts for the manuscript site.

Maps KG entity types to site categories:
  thinker   → "thinker"
  concept   → "concept"  (+ some become "framework", "diagnosis", "theological")
  tradition → "tradition"
  work      → "work"
  historical_period → "period"
  domain    → "domain"

Reads: knowledge-graph/backend/seeds/knowledge_base.json
Writes: manuscript-site/lib/graph-data.ts
"""

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KG_JSON = ROOT / "knowledge-graph" / "backend" / "seeds" / "knowledge_base.json"
OUTPUT = ROOT / "manuscript-site" / "lib" / "graph-data.ts"

# ── Chapter slug mapping ──────────────────────────────────────────
# Maps chapter numbers to slugs as they appear in the manuscript site
CHAPTER_SLUGS = {
    0: "00-introduction",
    1: "01-neurodivergence-and-consciousness",
    2: "02-normies-psychos-schizos",
    3: "03-the-prophetic-function",
    4: "04-the-psychoanalytic-tradition",
    5: "05-popper-and-falsifiability",
    6: "06-kuhn-and-paradigm-shifts",
    7: "07-complexity-science-and-emergence",
    8: "08-philosophy-of-ml-and-ai",
    9: "09-embeddings-transformers-blessing-of-dimensionality",
    10: "10-judea-pearl-and-causal-graphs",
    11: "11-the-hegel-popper-kuhn-pearl-synthesis",
    12: "12-contemporary-intellectual-movements",
    13: "13-old-testament-consciousness-emergence",
    14: "14-the-fall-as-felix-culpa",
    15: "15-new-testament-redemption",
    16: "16-trinity-as-strange-loop",
    17: "17-quran-and-christian-reconciliation",
    18: "18-all-trajectories-converge-world-wisdom",
    19: "19-the-cyclical-christ-samsara-nirvana",
    20: "20-the-riemann-sphere-theology",
    21: "21-antichrist-epstein-and-structural-evil",
    22: "22-sexual-liberation-and-embodied-theology",
    23: "23-aesthetics-and-the-theology-of-beauty",
    24: "24-republic-of-letters-historical",
    25: "25-republic-of-ai-agents",
    26: "26-free-will-determinism-plekhanov",
    27: "27-the-apostolic-task",
    28: "28-the-kirill-principle",
    29: "29-ecclesiology-community-structure",
    30: "30-the-development-lab-curriculum",
    31: "31-male-loneliness-crisis",
    32: "32-mental-health-crisis",
    33: "33-ai-safety-automation",
    34: "34-economic-inequality",
    35: "35-societal-polarization",
    36: "36-climate-crisis",
    37: "37-geopolitical-fragmentation",
    38: "38-the-meaning-crisis",
    39: "39-education-crisis",
}

# ── Category assignment ───────────────────────────────────────────
# Some concepts from KG should map to specific site categories
CONCEPT_CATEGORY_OVERRIDES = {
    # Frameworks (methodological/analytical tools)
    "concept-riemann-sphere-theology": "framework",
    "concept-pearl-hierarchy": "framework",
    "concept-ooda-loop": "framework",
    "concept-dialectical-engine": "framework",
    "concept-strange-loop": "framework",
    "concept-active-inference": "framework",
    "concept-republic-of-ai-agents": "framework",
    "concept-samsaric-cycle": "framework",
    "concept-flow-state": "framework",
    "concept-heros-journey": "framework",
    "concept-expanding-circle": "framework",
    "concept-mimetic-desire": "framework",
    "concept-scapegoat-mechanism": "framework",
    "concept-panopticon": "framework",
    "concept-antifragility": "framework",
    "concept-black-swan": "framework",
    "concept-system-1-2": "framework",
    "concept-hegel-popper-kuhn-pearl": "framework",
    "concept-ifs": "framework",
    "concept-backdoor-criterion": "framework",
    "concept-do-calculus": "framework",
    "concept-kirill-function": "framework",
    "concept-kirill-test": "framework",
    "concept-kirill-principle": "framework",
    # Diagnostic (social taxonomy)
    "concept-normie-psycho-schizo": "diagnosis",
    "concept-prophetic-function": "diagnosis",
    "concept-banality-of-evil": "diagnosis",
    "concept-mauvaise-foi": "diagnosis",
    "concept-das-man": "diagnosis",
    "concept-burnout-society": "diagnosis",
    "concept-somatic-critique": "diagnosis",
    "concept-moloch": "diagnosis",
    # Theological (metaphysical/spiritual)
    "concept-trinity-strange-loop": "theological",
    "concept-felix-culpa": "theological",
    "concept-point-at-infinity": "theological",
    "concept-removable-singularity": "theological",
    "concept-apostolic-realization": "theological",
    "concept-theosis": "theological",
    "concept-wahdat-al-wujud": "theological",
    "concept-logoi": "theological",
    "concept-bodhisattva": "theological",
    "concept-brahman-atman": "theological",
    "concept-wu-wei": "theological",
    "concept-sunyata": "theological",
    "concept-tawhid": "theological",
    "concept-deus-sive-natura": "theological",
    "concept-omnipresence-correction": "theological",
    "concept-positive-derivative": "theological",
    "concept-torus-objection": "theological",
    "concept-genus-raising": "theological",
    "concept-hyperbolic-divergence": "theological",
    "concept-dancing-and-joy": "theological",
    "concept-ubuntu-philosophy": "theological",
    # Crisis (societal problems addressed in Part 5)
    "concept-alignment-problem": "crisis",
    "concept-problems-of-success": "crisis",
}

# Core nodes that should appear in "core" graph view
CORE_IDS = {
    # Core thinkers (the synthesis pillars + major contributors)
    "thinker-hegel", "thinker-popper", "thinker-kuhn", "thinker-pearl",
    "thinker-boyd", "thinker-hofstadter", "thinker-girard", "thinker-friston",
    "thinker-aristotle", "thinker-nietzsche", "thinker-dostoevsky",
    "thinker-camus", "thinker-taleb", "thinker-arendt", "thinker-weil",
    "thinker-pirsig", "thinker-campbell", "thinker-plekhanov",
    "thinker-vervaeke", "thinker-mcgilchrist", "thinker-jaynes", "thinker-milton",
    # Core frameworks
    "concept-riemann-sphere-theology", "concept-pearl-hierarchy", "concept-ooda-loop",
    "concept-dialectical-engine", "concept-strange-loop", "concept-active-inference",
    "concept-republic-of-ai-agents", "concept-samsaric-cycle",
    "concept-hegel-popper-kuhn-pearl", "concept-heros-journey",
    # Core concepts
    "concept-falsifiability", "concept-paradigm-shift", "concept-emergence",
    "concept-plekhanov-synthesis", "concept-phase-transition",
    "concept-word-embeddings", "concept-flow-state",
    "concept-goedel-incompleteness", "concept-quality",
    # Core diagnostic
    "concept-normie-psycho-schizo", "concept-prophetic-function",
    # Core theological
    "concept-trinity-strange-loop", "concept-felix-culpa",
    "concept-point-at-infinity", "concept-removable-singularity",
    "concept-apostolic-realization", "concept-brahman-atman",
    "concept-omnipresence-correction", "concept-positive-derivative",
    "concept-genus-raising", "concept-torus-objection",
    # Core crises
    "concept-alignment-problem", "concept-problems-of-success",
}


def chapter_refs_to_slugs(refs: list) -> list[str]:
    """Convert manuscript_refs to chapter slug strings."""
    slugs = []
    for ref in refs:
        ch = ref.get("chapter")
        if ch is not None and ch in CHAPTER_SLUGS:
            slug = CHAPTER_SLUGS[ch]
            if slug not in slugs:
                slugs.append(slug)
    return slugs


def make_short_id(kg_id: str) -> str:
    """Convert KG ID to shorter site ID.
    e.g. 'thinker-karl-popper' -> 'popper', 'concept-falsifiability' -> 'falsifiability'
    """
    # Remove prefix
    for prefix in ["thinker-", "concept-", "tradition-", "work-", "period-", "domain-"]:
        if kg_id.startswith(prefix):
            return kg_id[len(prefix):]
    return kg_id


def make_label(name: str) -> str:
    """Clean up label for display. Max ~30 chars."""
    if len(name) > 35:
        return name[:32] + "..."
    return name


def truncate_desc(desc: str, max_len: int = 150) -> str:
    """Truncate description for tooltip display."""
    if not desc:
        return ""
    desc = desc.strip()
    if len(desc) <= max_len:
        return desc
    return desc[:max_len-3].rsplit(" ", 1)[0] + "..."


def get_concept_category(concept_id: str) -> str:
    """Determine site category for a KG concept."""
    if concept_id in CONCEPT_CATEGORY_OVERRIDES:
        return CONCEPT_CATEGORY_OVERRIDES[concept_id]
    return "concept"


def escape_ts(s: str) -> str:
    """Escape string for TypeScript string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", "")


# ── Edge type mapping ─────────────────────────────────────────────
# KG relation types → site edge types
CONCEPT_REL_MAP = {
    "extends": "builds_on",
    "contradicts": "critiques",
    "synthesizes": "synthesizes",
    "precedes": "builds_on",
    "formalizes": "formalizes",
    "applies": "applies_to",
    "analogous_to": "synthesizes",
}

THINKER_REL_MAP = {
    "influenced": "builds_on",
    "student_of": "builds_on",
    "contemporary_of": "synthesizes",
    "opposed": "critiques",
    "collaborated": "synthesizes",
}


def main():
    with open(KG_JSON, "r") as f:
        kg = json.load(f)

    nodes = []
    edges = []
    id_map = {}  # kg_id -> site_id
    used_ids = set()

    # ── Process Thinkers ──────────────────────────────────────
    thinkers = kg.get("thinkers", {})
    for kg_id, t in thinkers.items():
        site_id = make_short_id(kg_id)
        # Handle collisions
        if site_id in used_ids:
            site_id = kg_id
        used_ids.add(site_id)
        id_map[kg_id] = site_id

        chapters = chapter_refs_to_slugs(t.get("manuscript_refs", []))
        bio = truncate_desc(t.get("bio", ""), 200)
        name = t.get("name", "")
        # Use last name for label if it's a person
        parts = name.split()
        label = parts[-1] if len(parts) > 1 else name
        # Some thinkers are known by full name
        if label in ("Arabi", "Khaldun", "Beauvoir", "Nhat", "Han"):
            if len(parts) >= 2:
                label = " ".join(parts[-2:])

        is_core = kg_id in CORE_IDS
        tier = t.get("tier")
        era = t.get("era", "")

        node = {
            "id": site_id,
            "label": make_label(label),
            "category": "thinker",
            "chapters": chapters,
            "description": escape_ts(bio),
            "core": is_core,
        }
        # Add extra metadata
        if tier is not None:
            node["tier"] = tier
        if era:
            node["era"] = era
        nodes.append(node)

    # ── Process Concepts ──────────────────────────────────────
    concepts = kg.get("concepts", {})
    for kg_id, c in concepts.items():
        site_id = make_short_id(kg_id)
        if site_id in used_ids:
            site_id = kg_id
        used_ids.add(site_id)
        id_map[kg_id] = site_id

        chapters = chapter_refs_to_slugs(c.get("manuscript_refs", []))
        desc = truncate_desc(c.get("description", ""), 200)
        cat = get_concept_category(kg_id)
        is_core = kg_id in CORE_IDS

        node = {
            "id": site_id,
            "label": make_label(c.get("name", "")),
            "category": cat,
            "chapters": chapters,
            "description": escape_ts(desc),
            "core": is_core,
        }
        claim = c.get("claim_status")
        if claim:
            node["claimStatus"] = claim
        pearl = c.get("pearl_level")
        if pearl:
            node["pearlLevel"] = pearl
        nodes.append(node)

    # ── Process Traditions ────────────────────────────────────
    traditions = kg.get("traditions", {})
    for kg_id, t in traditions.items():
        site_id = make_short_id(kg_id)
        if site_id in used_ids:
            site_id = kg_id
        used_ids.add(site_id)
        id_map[kg_id] = site_id

        chapters = chapter_refs_to_slugs(t.get("manuscript_refs", []))
        desc = truncate_desc(t.get("description", ""), 200)

        nodes.append({
            "id": site_id,
            "label": make_label(t.get("name", "")),
            "category": "tradition",
            "chapters": chapters,
            "description": escape_ts(desc),
            "core": False,
        })

    # ── Process Works ─────────────────────────────────────────
    works = kg.get("works", {})
    for kg_id, w in works.items():
        site_id = make_short_id(kg_id)
        if site_id in used_ids:
            site_id = kg_id
        used_ids.add(site_id)
        id_map[kg_id] = site_id

        chapters = chapter_refs_to_slugs(w.get("manuscript_refs", []))
        title = w.get("title", "")
        year = w.get("year")
        desc_parts = []
        if year:
            desc_parts.append(f"({year})")
        desc = " ".join(desc_parts) if desc_parts else ""

        nodes.append({
            "id": site_id,
            "label": make_label(title),
            "category": "work",
            "chapters": chapters,
            "description": escape_ts(desc),
            "core": False,
        })

    # ── Process Historical Periods ────────────────────────────
    periods = kg.get("historical_periods", {})
    for kg_id, p in periods.items():
        site_id = make_short_id(kg_id)
        if site_id in used_ids:
            site_id = kg_id
        used_ids.add(site_id)
        id_map[kg_id] = site_id

        chapters = chapter_refs_to_slugs(p.get("manuscript_refs", []))
        desc = truncate_desc(p.get("description", ""), 200)
        cycle = p.get("cycle_phase")

        node = {
            "id": site_id,
            "label": make_label(p.get("name", "")),
            "category": "period",
            "chapters": chapters,
            "description": escape_ts(desc),
            "core": False,
        }
        if cycle:
            node["cyclePhase"] = cycle
        nodes.append(node)

    # ── Process Domains ───────────────────────────────────────
    domains = kg.get("domains", {})
    for kg_id, d in domains.items():
        site_id = make_short_id(kg_id)
        if site_id in used_ids:
            site_id = kg_id
        used_ids.add(site_id)
        id_map[kg_id] = site_id

        chapters = chapter_refs_to_slugs(d.get("manuscript_refs", []))
        desc = truncate_desc(d.get("description", ""), 200)

        nodes.append({
            "id": site_id,
            "label": make_label(d.get("name", "")),
            "category": "domain",
            "chapters": chapters,
            "description": escape_ts(desc),
            "core": False,
        })

    # ── Build Edges ───────────────────────────────────────────
    seen_edges = set()

    def add_edge(source_kg_id, target_kg_id, edge_type, strength=0.7, label=""):
        src = id_map.get(source_kg_id)
        tgt = id_map.get(target_kg_id)
        if not src or not tgt:
            return
        if src == tgt:
            return
        key = f"{src}-{tgt}-{edge_type}"
        if key in seen_edges:
            return
        seen_edges.add(key)
        edge = {
            "source": src,
            "target": tgt,
            "type": edge_type,
            "strength": round(strength, 2),
        }
        if label:
            edge["label"] = escape_ts(label)
        edges.append(edge)

    # Thinker-to-thinker relations
    for kg_id, t in thinkers.items():
        for rel in t.get("related_thinkers", []):
            target = rel.get("target_id")
            rel_type = THINKER_REL_MAP.get(rel.get("relation_type"), "builds_on")
            desc = rel.get("description", "")
            add_edge(kg_id, target, rel_type, 0.7, truncate_desc(desc, 60))

    # Concept-to-concept relations
    for kg_id, c in concepts.items():
        for rel in c.get("related_concepts", []):
            target = rel.get("target_id")
            rel_type = CONCEPT_REL_MAP.get(rel.get("relation_type"), "builds_on")
            desc = rel.get("description", "")
            add_edge(kg_id, target, rel_type, 0.75, truncate_desc(desc, 60))

        # Concept-to-originator
        originator = c.get("originator_id")
        if originator:
            add_edge(originator, kg_id, "formalizes", 0.85)

        # Concept-to-developers
        for dev_id in c.get("developer_ids", []):
            add_edge(dev_id, kg_id, "builds_on", 0.65)

    # Thinker-to-tradition
    for kg_id, t in thinkers.items():
        for trad_id in t.get("traditions", []):
            add_edge(kg_id, trad_id, "builds_on", 0.5)

    # Tradition parent/child
    for kg_id, t in traditions.items():
        for parent_id in t.get("parent_tradition_ids", []):
            add_edge(parent_id, kg_id, "builds_on", 0.6)

    # Work-to-author
    for kg_id, w in works.items():
        for author_id in w.get("author_ids", []):
            add_edge(author_id, kg_id, "formalizes", 0.6)

    # Work-to-concepts
    for kg_id, w in works.items():
        for concept_id in w.get("concepts_introduced", []):
            add_edge(kg_id, concept_id, "formalizes", 0.7)
        for concept_id in w.get("concepts_developed", []):
            add_edge(kg_id, concept_id, "builds_on", 0.55)

    # Domain-to-thinker
    for kg_id, d in domains.items():
        for thinker_id in d.get("key_thinker_ids", []):
            add_edge(thinker_id, kg_id, "applies_to", 0.4)

    # ── Read graph.links for additional edges ────────────────
    # The KG stores edges both in nested entity data AND in graph.links.
    # graph.links captures relations the entity-level extraction misses,
    # especially domain/tradition membership and cross-type links.
    LINK_REL_MAP = {
        # Domain membership
        "works_in_domain": ("applies_to", 0.35),
        "in_domain": ("applies_to", 0.35),
        "subdomain_of": ("builds_on", 0.5),
        # Tradition membership
        "belongs_to_tradition": ("builds_on", 0.45),
        "part_of_tradition": ("builds_on", 0.45),
        "child_of_tradition": ("builds_on", 0.55),
        # Work relations
        "authored_work": ("formalizes", 0.6),
        "introduces_concept": ("formalizes", 0.7),
        # Thinker relations
        "influenced": ("builds_on", 0.7),
        "student_of": ("builds_on", 0.7),
        "contemporary_of": ("synthesizes", 0.4),
        "opposed": ("critiques", 0.7),
        "collaborated": ("synthesizes", 0.5),
        # Concept relations
        "originated_concept": ("formalizes", 0.85),
        "developed_concept": ("builds_on", 0.65),
        "develops_concept": ("builds_on", 0.65),
        "extends_concept": ("builds_on", 0.75),
        "analogous_to_concept": ("synthesizes", 0.6),
        "applies_concept": ("applies_to", 0.6),
        "precedes_concept": ("builds_on", 0.6),
        "synthesizes_concept": ("synthesizes", 0.75),
        "contradicts_concept": ("critiques", 0.75),
        "formalizes_concept": ("formalizes", 0.75),
    }

    graph_links = kg.get("graph", {}).get("links", [])
    for link in graph_links:
        rel = link.get("relation", "")
        source_kg = link.get("source", "")
        target_kg = link.get("target", "")
        if rel in LINK_REL_MAP:
            edge_type, strength = LINK_REL_MAP[rel]
            add_edge(source_kg, target_kg, edge_type, strength)

    # ── Add crisis nodes from Part 5 chapters (not in KG as concepts) ──
    CRISIS_NODES = [
        ("crisis-male-loneliness", "Male Loneliness", "Missing synthesis between traditional patriarchy and anomic individualism; manosphere as psycho-class capture of genuine male suffering", ["31-male-loneliness-crisis"]),
        ("crisis-mental-health", "Mental Health Crisis", "Normal responses to abnormal conditions pathologized; pharmaceutical capture of genuine suffering", ["32-mental-health-crisis"]),
        ("crisis-ai-safety", "AI Safety", "AI as both most powerful psycho-class tool and most powerful prophetic instrument; alignment as theological problem", ["33-ai-safety-automation"]),
        ("crisis-inequality", "Economic Inequality", "Information asymmetry as primary extraction mechanism; complexity as weapon; psycho-class capture of financial systems", ["34-economic-inequality"]),
        ("crisis-polarization", "Societal Polarization", "Gutenberg parallel: destroyed epistemic infrastructure without replacement; 21st-century Wars of Religion", ["35-societal-polarization"]),
        ("crisis-climate", "Climate Crisis", "System of systems requiring causal ML; fossil fuel psycho-class denial infrastructure", ["36-climate-crisis"]),
        ("crisis-geopolitics", "Geopolitical Fragmentation", "Post-WWII order collapsing; thesis generating its own antithesis; Ukraine and Lebanon as lived examples", ["37-geopolitical-fragmentation"]),
        ("crisis-meaning", "Meaning Crisis", "Meta-crisis: modernity killed God, postmodernity killed grand narratives, nothing replaced them", ["38-the-meaning-crisis"]),
        ("crisis-education", "Education Crisis", "Credential-knowledge decoupling; universities as gatekeepers rather than knowledge generators", ["39-education-crisis"]),
    ]

    for cid, label, desc, chapters in CRISIS_NODES:
        if cid not in used_ids:
            used_ids.add(cid)
            id_map[cid] = cid  # Register in id_map so add_edge can resolve
            is_core = cid in {"crisis-meaning", "crisis-ai-safety", "crisis-male-loneliness", "crisis-inequality"}
            nodes.append({
                "id": cid,
                "label": label,
                "category": "crisis",
                "chapters": chapters,
                "description": escape_ts(desc),
                "core": is_core,
            })

    # Add edges from meaning crisis to other crises
    for cid, _, _, _ in CRISIS_NODES:
        if cid != "crisis-meaning" and cid in used_ids:
            add_edge_raw = lambda s, t, et, st=0.7, lb="": edges.append({"source": s, "target": t, "type": et, "strength": round(st, 2), **({"label": lb} if lb else {})})
    # meaning crisis → other crises
    for cid, _, _, _ in CRISIS_NODES:
        if cid != "crisis-meaning":
            edges.append({"source": "crisis-meaning", "target": cid, "type": "builds_on", "strength": 0.7})
    # republic model → crises
    if "republic-of-ai-agents" in used_ids:
        for cid, _, _, _ in CRISIS_NODES:
            edges.append({"source": "republic-of-ai-agents", "target": cid, "type": "applies_to", "strength": 0.6})

    # ── Period chronological chain ────────────────────────────
    # Connect periods chronologically so they aren't isolated
    PERIOD_CHAIN = [
        "period-axial-age",
        "period-classical-antiquity",
        "period-late-antiquity",
        "period-medieval",
        "period-renaissance",
        "period-enlightenment",
        "period-industrial-revolution",
        "period-modernity",
        "period-cold-war",
        "period-information-age",
    ]
    for i in range(len(PERIOD_CHAIN) - 1):
        add_edge(PERIOD_CHAIN[i], PERIOD_CHAIN[i + 1], "builds_on", 0.4)

    # Connect a few isolated traditions/concepts to natural neighbors
    STRUCTURAL_EDGES = [
        ("tradition-confucianism", "domain-ethics", "applies_to", 0.4),
        ("tradition-confucianism", "period-axial-age", "applies_to", 0.3),
        ("tradition-critical-theory", "domain-political-philosophy", "applies_to", 0.4),
        ("tradition-empiricism", "domain-epistemology", "applies_to", 0.4),
        ("period-republic-of-letters", "period-enlightenment", "builds_on", 0.5),
        ("tradition-eacc", "concept-alignment-problem", "critiques", 0.6),
        ("period-metacrisis", "crisis-meaning", "builds_on", 0.7),
        ("period-islamic-golden-age", "period-medieval", "synthesizes", 0.4),
    ]
    for src, tgt, etype, strength in STRUCTURAL_EDGES:
        add_edge(src, tgt, etype, strength)

    # ── Generate TypeScript ───────────────────────────────────
    node_set = {n["id"] for n in nodes}

    # Filter edges to only include nodes that exist
    valid_edges = [e for e in edges if e["source"] in node_set and e["target"] in node_set]

    lines = []
    lines.append('import type { ConceptNode, ConceptEdge } from "@/types/graph";')
    lines.append("")
    lines.append("// Auto-generated from knowledge-graph/backend/seeds/knowledge_base.json")
    lines.append(f"// {len(nodes)} nodes, {len(valid_edges)} edges")
    lines.append(f"// Generated by scripts/kg-to-site.py")
    lines.append("")
    lines.append("export const NODES: ConceptNode[] = [")

    # Group nodes by category for readability
    categories_order = ["thinker", "framework", "concept", "diagnosis", "theological", "crisis", "tradition", "work", "period", "domain"]
    category_names = {
        "thinker": "Thinkers",
        "framework": "Frameworks",
        "concept": "Concepts",
        "diagnosis": "Diagnostic Categories",
        "theological": "Theological Concepts",
        "crisis": "Societal Crises",
        "tradition": "Traditions",
        "work": "Works",
        "period": "Historical Periods",
        "domain": "Domains",
    }

    for cat in categories_order:
        cat_nodes = [n for n in nodes if n["category"] == cat]
        if not cat_nodes:
            continue
        lines.append(f"  // ── {category_names.get(cat, cat)} ({len(cat_nodes)}) ──")
        for n in cat_nodes:
            chapters_str = json.dumps(n["chapters"])
            parts = [
                f'id: "{n["id"]}"',
                f'label: "{n["label"]}"',
                f'category: "{n["category"]}"',
                f"chapters: {chapters_str}",
            ]
            if n.get("description"):
                parts.append(f'description: "{n["description"]}"')
            if n.get("core"):
                parts.append("core: true")
            lines.append("  { " + ", ".join(parts) + " },")
    lines.append("];")
    lines.append("")

    # Edges
    lines.append("export const EDGES: ConceptEdge[] = [")
    for e in valid_edges:
        parts = [
            f'source: "{e["source"]}"',
            f'target: "{e["target"]}"',
            f'type: "{e["type"]}"',
            f'strength: {e["strength"]}',
        ]
        if e.get("label"):
            parts.append(f'label: "{e["label"]}"')
        lines.append("  { " + ", ".join(parts) + " },")
    lines.append("];")
    lines.append("")

    # Helper functions
    lines.append("// ── Helpers for graph mode filtering ─────────────────────")
    lines.append("")
    lines.append('const CORE_NODE_IDS = new Set(NODES.filter((n) => n.core).map((n) => n.id));')
    lines.append("")
    lines.append("export function getCoreNodes(): ConceptNode[] {")
    lines.append('  return NODES.filter((n) => n.core);')
    lines.append("}")
    lines.append("")
    lines.append("export function getCoreEdges(): ConceptEdge[] {")
    lines.append('  return EDGES.filter((e) => CORE_NODE_IDS.has(e.source) && CORE_NODE_IDS.has(e.target));')
    lines.append("}")
    lines.append("")
    lines.append("const INTELLECTUAL_CATEGORIES = new Set([")
    lines.append('  "thinker", "framework", "concept", "diagnosis", "theological", "crisis",')
    lines.append("]);")
    lines.append("const INTELLECTUAL_NODE_IDS = new Set(")
    lines.append('  NODES.filter((n) => INTELLECTUAL_CATEGORIES.has(n.category)).map((n) => n.id),')
    lines.append(");")
    lines.append("")
    lines.append("export function getIntellectualNodes(): ConceptNode[] {")
    lines.append('  return NODES.filter((n) => INTELLECTUAL_CATEGORIES.has(n.category));')
    lines.append("}")
    lines.append("")
    lines.append("export function getIntellectualEdges(): ConceptEdge[] {")
    lines.append("  return EDGES.filter(")
    lines.append('    (e) => INTELLECTUAL_NODE_IDS.has(e.source) && INTELLECTUAL_NODE_IDS.has(e.target),')
    lines.append("  );")
    lines.append("}")
    lines.append("")

    output_str = "\n".join(lines)
    with open(OUTPUT, "w") as f:
        f.write(output_str)

    print(f"Generated {OUTPUT}")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(valid_edges)}")
    print(f"  Categories: {', '.join(cat + '=' + str(len([n for n in nodes if n['category'] == cat])) for cat in categories_order if any(n['category'] == cat for n in nodes))}")


if __name__ == "__main__":
    main()
