"""Seed data for concepts — ~130 intellectual ideas spanning the manuscript's foundation.

Organized by domain with deterministic IDs (concept-{slug}).
Cross-references thinker IDs from thinkers.py and tradition/domain IDs from their
respective seed files.
"""

from graph.knowledge_entities import (
    Concept,
    ConceptRelation,
    ConceptRelationType,
    ManuscriptReference,
)

# Shorthand helpers
_R = ManuscriptReference
_CR = ConceptRelation
_T = ConceptRelationType


def create_concepts() -> list[Concept]:
    return [
        # =================================================================
        # EPISTEMOLOGY — The four-pillar synthesis
        # =================================================================

        Concept(
            id="concept-falsifiability",
            name="Falsifiability",
            description="A theory is scientific only if it specifies conditions under which it would be disproven. The discipline that prevents theology from becoming ideology.",
            domain_ids=["domain-epistemology", "domain-philosophy-of-science"],
            originator_id="thinker-popper",
            tradition_ids=["tradition-vienna-circle"],
            manuscript_refs=[_R(chapter=5, part="part2-epistemology", notes="Core Popperian principle")],
            related_concepts=[
                _CR(target_id="concept-paradigm-shift", relation_type=_T.SYNTHESIZES, description="Combined with Kuhn in Chapter 11 synthesis"),
                _CR(target_id="concept-historicism", relation_type=_T.CONTRADICTS, description="Popper's critique of unfalsifiable historical determinism"),
            ],
            pearl_level="counterfactual",
        ),
        Concept(
            id="concept-paradigm-shift",
            name="Paradigm Shift",
            description="Gestalt revolution in scientific understanding: anomalies accumulate until a new framework replaces the old, explaining both old successes and new anomalies.",
            domain_ids=["domain-epistemology", "domain-philosophy-of-science"],
            originator_id="thinker-kuhn",
            manuscript_refs=[_R(chapter=6, part="part2-epistemology")],
            related_concepts=[
                _CR(target_id="concept-normal-science", relation_type=_T.PRECEDES),
                _CR(target_id="concept-dialectical-engine", relation_type=_T.ANALOGOUS_TO, description="Kuhn's paradigm shifts = Boyd's destructive deduction + creative induction"),
            ],
        ),
        Concept(
            id="concept-normal-science",
            name="Normal Science",
            description="Puzzle-solving within an accepted paradigm. Productive phase that accumulates anomalies eventually triggering crisis.",
            domain_ids=["domain-philosophy-of-science"],
            originator_id="thinker-kuhn",
            manuscript_refs=[_R(chapter=6, part="part2-epistemology")],
        ),
        Concept(
            id="concept-do-calculus",
            name="Do-Calculus",
            description="Formal system for computing interventional probabilities from observational data. The mathematical formalization of the prophetic function.",
            domain_ids=["domain-causal-inference", "domain-statistics"],
            originator_id="thinker-pearl",
            manuscript_refs=[_R(chapter=10, part="part2-epistemology", notes="Prophetic seeing formalized")],
            related_concepts=[
                _CR(target_id="concept-pearl-hierarchy", relation_type=_T.FORMALIZES),
                _CR(target_id="concept-backdoor-criterion", relation_type=_T.PRECEDES),
            ],
            pearl_level="intervention",
        ),
        Concept(
            id="concept-pearl-hierarchy",
            name="Pearl's Causal Hierarchy",
            description="Three levels of causal reasoning: association (seeing), intervention (doing), counterfactual (imagining). Maps onto normie/psycho/schizo perception levels.",
            domain_ids=["domain-causal-inference", "domain-epistemology"],
            originator_id="thinker-pearl",
            manuscript_refs=[_R(chapter=10, part="part2-epistemology")],
            related_concepts=[
                _CR(target_id="concept-normie-psycho-schizo", relation_type=_T.ANALOGOUS_TO, description="Pearl's hierarchy maps onto social taxonomy"),
            ],
            pearl_level="counterfactual",
        ),
        Concept(
            id="concept-backdoor-criterion",
            name="Backdoor Criterion",
            description="Sufficient conditions for identifying causal effects by conditioning on a set of variables that blocks all backdoor paths.",
            domain_ids=["domain-causal-inference"],
            originator_id="thinker-pearl",
            manuscript_refs=[_R(chapter=10, part="part2-epistemology")],
            pearl_level="intervention",
        ),
        Concept(
            id="concept-dialectical-method",
            name="Dialectical Method (Thesis-Antithesis-Synthesis)",
            description="Hegel's pattern of intellectual development: a position generates its own negation, and the tension is resolved at a higher level.",
            domain_ids=["domain-epistemology", "domain-metaphysics"],
            originator_id="thinker-hegel",
            tradition_ids=["tradition-german-idealism"],
            manuscript_refs=[_R(chapter=11, part="part2-epistemology", notes="The PATTERN of the theology")],
            related_concepts=[
                _CR(target_id="concept-dialectical-engine", relation_type=_T.PRECEDES, description="Boyd formalized the dialectical method"),
                _CR(target_id="concept-historicism", relation_type=_T.PRECEDES),
            ],
        ),
        Concept(
            id="concept-historicism",
            name="Historicism",
            description="The belief that history follows necessary laws and has a predetermined direction. Popper's primary target in The Open Society.",
            domain_ids=["domain-political-philosophy", "domain-epistemology"],
            originator_id="thinker-hegel",
            developer_ids=["thinker-marx"],
            manuscript_refs=[_R(chapter=5, part="part2-epistemology", notes="Popper's critique")],
            related_concepts=[
                _CR(target_id="concept-falsifiability", relation_type=_T.CONTRADICTS, description="Historicism is unfalsifiable by nature"),
            ],
        ),
        Concept(
            id="concept-hegel-popper-kuhn-pearl",
            name="Hegel-Popper-Kuhn-Pearl Synthesis",
            description="The key epistemological engine: Hegel gives the pattern, Popper the discipline, Kuhn the sociology, Pearl the methodology.",
            domain_ids=["domain-epistemology"],
            developer_ids=["thinker-hegel", "thinker-popper", "thinker-kuhn", "thinker-pearl"],
            manuscript_refs=[_R(chapter=11, part="part2-epistemology", notes="THE KEY CHAPTER")],
            related_concepts=[
                _CR(target_id="concept-dialectical-method", relation_type=_T.SYNTHESIZES),
                _CR(target_id="concept-falsifiability", relation_type=_T.SYNTHESIZES),
                _CR(target_id="concept-paradigm-shift", relation_type=_T.SYNTHESIZES),
                _CR(target_id="concept-do-calculus", relation_type=_T.SYNTHESIZES),
            ],
        ),

        # =================================================================
        # METAPHYSICS & THEOLOGY — Riemann sphere and Trinity
        # =================================================================

        Concept(
            id="concept-strange-loop",
            name="Strange Loop",
            description="A self-referential system where moving through a hierarchy brings you back to the starting point at a different level. Consciousness as strange loop.",
            domain_ids=["domain-logic", "domain-philosophy-of-mind"],
            originator_id="thinker-hofstadter",
            manuscript_refs=[_R(chapter=16, part="part3-metaphysics", notes="Trinity as strange loop")],
            related_concepts=[
                _CR(target_id="concept-goedel-incompleteness", relation_type=_T.APPLIES),
                _CR(target_id="concept-trinity-strange-loop", relation_type=_T.PRECEDES),
            ],
        ),
        Concept(
            id="concept-goedel-incompleteness",
            name="Godel's Incompleteness Theorems",
            description="Any sufficiently powerful formal system is incomplete: it contains truths it cannot prove from within itself. God as the Godelian truth of the universe.",
            domain_ids=["domain-mathematical-logic", "domain-philosophy"],
            originator_id="thinker-goedel",
            manuscript_refs=[_R(chapter=16, part="part3-metaphysics")],
            related_concepts=[
                _CR(target_id="concept-strange-loop", relation_type=_T.PRECEDES),
                _CR(target_id="concept-dialectical-engine", relation_type=_T.ANALOGOUS_TO, description="Boyd explicitly grounds Dialectic Engine in Godel"),
            ],
        ),
        Concept(
            id="concept-riemann-sphere-theology",
            name="Riemann Sphere Theology",
            description="God is the entire Riemann sphere (omnipresence). The point at infinity is how finite beings EXPERIENCE God's infinity. History is trajectory on the sphere.",
            domain_ids=["domain-metaphysics", "domain-systematic-theology", "domain-topology"],
            manuscript_refs=[_R(chapter=20, part="part3-metaphysics", notes="MATHEMATICAL CORE — with omnipresence correction")],
            related_concepts=[
                _CR(target_id="concept-point-at-infinity", relation_type=_T.FORMALIZES),
                _CR(target_id="concept-torus-objection", relation_type=_T.CONTRADICTS, description="The torus theology as alternative topology"),
                _CR(target_id="concept-genus-raising", relation_type=_T.EXTENDS),
                _CR(target_id="concept-omnipresence-correction", relation_type=_T.EXTENDS),
            ],
        ),
        Concept(
            id="concept-point-at-infinity",
            name="Point at Infinity",
            description="The single point compactifying the complex plane into the Riemann sphere. How finite beings experience God's transcendence.",
            domain_ids=["domain-topology", "domain-metaphysics"],
            manuscript_refs=[_R(chapter=20, part="part3-metaphysics")],
        ),
        Concept(
            id="concept-omnipresence-correction",
            name="Omnipresence Correction (from Advaita Vedanta)",
            description="God is not AT the point at infinity but IS the entire sphere. The approach to infinity is deepening awareness of omnipresent God, not travel toward distant one.",
            domain_ids=["domain-metaphysics", "domain-systematic-theology"],
            tradition_ids=["tradition-advaita-vedanta"],
            manuscript_refs=[_R(chapter=20, part="part3-metaphysics", notes="FUNDAMENTAL REVISION from Hindu reviewer")],
            related_concepts=[
                _CR(target_id="concept-brahman-atman", relation_type=_T.SYNTHESIZES, description="Hindu non-dualism applied to Riemann sphere"),
                _CR(target_id="concept-riemann-sphere-theology", relation_type=_T.EXTENDS),
            ],
        ),
        Concept(
            id="concept-trinity-strange-loop",
            name="Trinity as Strange Loop",
            description="Father=generative ground (formal system + loving source), Son=self-expression (Godelian self-reference + image), Spirit=process of self-reference (love between them).",
            domain_ids=["domain-systematic-theology", "domain-logic"],
            developer_ids=["thinker-hofstadter"],
            tradition_ids=["tradition-christianity"],
            manuscript_refs=[_R(chapter=16, part="part3-metaphysics", notes="With Morskie Kotiki correction: must include LOVE")],
            related_concepts=[
                _CR(target_id="concept-strange-loop", relation_type=_T.APPLIES),
                _CR(target_id="concept-goedel-incompleteness", relation_type=_T.APPLIES),
            ],
        ),
        Concept(
            id="concept-felix-culpa",
            name="Felix Culpa (Fortunate Fall)",
            description="The Fall is the initialization of the trajectory — you cannot approach infinity from the origin without first moving away. The Fall is simultaneously loss and gain.",
            domain_ids=["domain-systematic-theology", "domain-metaphysics"],
            tradition_ids=["tradition-christianity"],
            manuscript_refs=[_R(chapter=14, part="part3-metaphysics")],
            related_concepts=[
                _CR(target_id="concept-riemann-sphere-theology", relation_type=_T.APPLIES),
            ],
        ),
        Concept(
            id="concept-apostolic-realization",
            name="Apostolic Realization Thesis",
            description="The Logos of Christ was not a pre-existing fact passively discovered but something that EMERGED through the apostles' collective experience. Jesus=local interaction, apostles=network, Logos=emergent property.",
            domain_ids=["domain-systematic-theology"],
            tradition_ids=["tradition-christianity"],
            manuscript_refs=[_R(chapter=15, part="part3-metaphysics")],
            related_concepts=[
                _CR(target_id="concept-emergence", relation_type=_T.APPLIES),
            ],
        ),
        Concept(
            id="concept-samsaric-cycle",
            name="Samsaric Cycle / Cyclical Christ",
            description="Each epoch repeats: antichrist capture -> prophetic vision -> crucifixion -> resurrection -> pentecost -> samsaric turn. The spiral ascends.",
            domain_ids=["domain-systematic-theology", "domain-metaphysics"],
            tradition_ids=["tradition-buddhism", "tradition-christianity"],
            manuscript_refs=[_R(chapter=19, part="part3-metaphysics")],
            related_concepts=[
                _CR(target_id="concept-dialectical-method", relation_type=_T.ANALOGOUS_TO, description="Dialectical spiral in theological register"),
                _CR(target_id="concept-paradigm-shift", relation_type=_T.ANALOGOUS_TO),
            ],
        ),
        Concept(
            id="concept-torus-objection",
            name="Torus Objection",
            description="Why not a torus (genus 1) instead of a sphere? No point at infinity, pure immanence, paths return. The theological topology is a CHOICE, not a necessity.",
            domain_ids=["domain-topology", "domain-metaphysics"],
            manuscript_refs=[_R(chapter=20, part="part3-metaphysics", notes="Appendix A.2")],
            related_concepts=[
                _CR(target_id="concept-riemann-sphere-theology", relation_type=_T.CONTRADICTS),
                _CR(target_id="concept-genus-raising", relation_type=_T.PRECEDES),
            ],
        ),
        Concept(
            id="concept-genus-raising",
            name="Genus-Raising",
            description="Each epoch's paradigm shift = genus increase of Riemann surface. Genus 0=pre-modern, 1=modern/secular, 2+=post-singularity. Enlightenment = topological phase transition.",
            domain_ids=["domain-topology", "domain-metaphysics"],
            manuscript_refs=[_R(chapter=20, part="part3-metaphysics", notes="Appendix A.2 — from anonymous topological critic")],
            related_concepts=[
                _CR(target_id="concept-torus-objection", relation_type=_T.EXTENDS),
                _CR(target_id="concept-hyperbolic-divergence", relation_type=_T.PRECEDES),
            ],
        ),
        Concept(
            id="concept-hyperbolic-divergence",
            name="Hyperbolic Divergence (Ruslan's Critique)",
            description="At genus >= 2, geodesics diverge exponentially. AGI may create curvature transition where micro-differences in AI values produce parallel civilizations.",
            domain_ids=["domain-topology", "domain-metaphysics", "domain-ai-ml"],
            manuscript_refs=[_R(chapter=33, part="part5-apostolic-agenda", notes="Ruslan's devastating topological critique")],
            related_concepts=[
                _CR(target_id="concept-genus-raising", relation_type=_T.EXTENDS),
                _CR(target_id="concept-riemann-sphere-theology", relation_type=_T.CONTRADICTS, description="If post-AGI topology is hyperbolic, convergence fails"),
            ],
        ),
        Concept(
            id="concept-removable-singularity",
            name="Removable Singularity (Crucifixion/Resurrection)",
            description="Point where function appears discontinuous from one perspective but is smooth from a higher-dimensional view. Crucifixion looks singular; resurrection reveals continuity.",
            domain_ids=["domain-topology", "domain-systematic-theology"],
            manuscript_refs=[_R(chapter=15, part="part3-metaphysics")],
        ),

        # =================================================================
        # PSYCHOLOGY & COGNITIVE SCIENCE
        # =================================================================

        Concept(
            id="concept-normie-psycho-schizo",
            name="Normie / Psycho / Schizo Taxonomy",
            description="Social ecology: normies (prosocial majority), psychos (dark triad optimized for manipulation), schizos (pattern recognition unconstrained by social consensus). Modes of cognition, not types of person.",
            domain_ids=["domain-psychology", "domain-sociology"],
            manuscript_refs=[_R(chapter=2, part="part1-psychology", notes="Core social taxonomy — with Solzhenitsyn caveat")],
            related_concepts=[
                _CR(target_id="concept-pearl-hierarchy", relation_type=_T.ANALOGOUS_TO),
                _CR(target_id="concept-banality-of-evil", relation_type=_T.EXTENDS, description="Arendt's correction: evil from normies who stop thinking"),
                _CR(target_id="concept-mimetic-desire", relation_type=_T.EXTENDS, description="Girard: normies absorb desires from social models"),
            ],
        ),
        Concept(
            id="concept-prophetic-function",
            name="Prophetic Function",
            description="Institutional space for non-rational cognition: shamans, prophets, holy fools. Modernity dismantled every container for prophetic perception while optimizing for psychopathic predation.",
            domain_ids=["domain-psychology", "domain-theology"],
            manuscript_refs=[_R(chapter=3, part="part1-psychology")],
            related_concepts=[
                _CR(target_id="concept-normie-psycho-schizo", relation_type=_T.EXTENDS),
                _CR(target_id="concept-flow-state", relation_type=_T.ANALOGOUS_TO, description="Prophetic state as flow with attenuated self-model"),
            ],
        ),
        Concept(
            id="concept-flow-state",
            name="Flow State",
            description="Optimal performance with paradoxical features: loss of self-awareness yet heightened agency. Explained through active inference — epistemic self-model attenuates.",
            domain_ids=["domain-cognitive-psychology", "domain-neuroscience"],
            originator_id="thinker-csikszentmihalyi",
            developer_ids=["thinker-kotler", "thinker-friston"],
            manuscript_refs=[_R(chapter=9, part="part2-epistemology", notes="Active inference account of flow")],
            related_concepts=[
                _CR(target_id="concept-active-inference", relation_type=_T.APPLIES),
                _CR(target_id="concept-prophetic-function", relation_type=_T.ANALOGOUS_TO),
            ],
        ),
        Concept(
            id="concept-active-inference",
            name="Active Inference / Free Energy Principle",
            description="Action and perception as forms of Bayesian inference minimizing variational free energy. The unifying mathematical framework for consciousness, flow, and the theology's epistemology.",
            domain_ids=["domain-neuroscience", "domain-philosophy-of-mind", "domain-ai-ml"],
            originator_id="thinker-friston",
            manuscript_refs=[_R(chapter=9, part="part2-epistemology")],
            related_concepts=[
                _CR(target_id="concept-flow-state", relation_type=_T.FORMALIZES),
                _CR(target_id="concept-dialectical-engine", relation_type=_T.ANALOGOUS_TO, description="Free energy minimization IS the dialectical engine"),
            ],
        ),
        Concept(
            id="concept-bicameral-mind",
            name="Bicameral Mind",
            description="Jaynes's hypothesis: human self-awareness emerged roughly 3000-1000 BCE. Hebrew biblical chronology is RIGHT about the age of experienced self-awareness.",
            domain_ids=["domain-psychology", "domain-history"],
            originator_id="thinker-jaynes",
            manuscript_refs=[_R(chapter=13, part="part3-metaphysics")],
        ),
        Concept(
            id="concept-system-1-2",
            name="System 1 / System 2 (+ System 3)",
            description="Kahneman's dual-process theory. System 1=fast/intuitive, System 2=slow/deliberate. Theology adds System 3: the strange loop's capacity for Godelian self-transcendence.",
            domain_ids=["domain-cognitive-psychology"],
            originator_id="thinker-kahneman",
            manuscript_refs=[_R(chapter=12, part="part2-epistemology", notes="Behavioral economics + theology's System 3")],
        ),
        Concept(
            id="concept-unconscious",
            name="The Unconscious",
            description="Freud's discovery of mental processes inaccessible to awareness. The topographic model maps onto Pearl's hierarchy; repetition compulsion = individual samsaric cycle.",
            domain_ids=["domain-psychoanalysis"],
            originator_id="thinker-freud",
            tradition_ids=["tradition-psychoanalysis"],
            manuscript_refs=[_R(chapter=4, part="part1-psychology")],
        ),
        Concept(
            id="concept-collective-unconscious",
            name="Collective Unconscious / Archetypes",
            description="Jung: shared embedding space of the human psyche. Archetypes = high-dimensional attractors. Individuation = personal Riemann sphere trajectory.",
            domain_ids=["domain-psychoanalysis", "domain-philosophy-of-mind"],
            originator_id="thinker-jung",
            tradition_ids=["tradition-jungian"],
            manuscript_refs=[_R(chapter=4, part="part1-psychology")],
            related_concepts=[
                _CR(target_id="concept-unconscious", relation_type=_T.EXTENDS),
            ],
        ),
        Concept(
            id="concept-paradoxical-theory-of-change",
            name="Paradoxical Theory of Change",
            description="Gestalt: change occurs when one becomes what one IS, not what one is not. Corrects theology's potential perpetual-dissatisfaction problem.",
            domain_ids=["domain-clinical-psychology"],
            originator_id="thinker-perls",
            tradition_ids=["tradition-gestalt"],
            manuscript_refs=[_R(chapter=4, part="part1-psychology")],
        ),
        Concept(
            id="concept-ifs",
            name="Internal Family Systems (Psyche as Republic)",
            description="Sub-personalities (managers, exiles, firefighters) governed by a Self. The psyche IS a republic of sub-personalities — the Republic model at individual scale.",
            domain_ids=["domain-clinical-psychology"],
            originator_id="thinker-schwartz",
            manuscript_refs=[_R(chapter=4, part="part1-psychology")],
        ),
        Concept(
            id="concept-left-right-hemisphere",
            name="Left/Right Hemisphere Thesis",
            description="McGilchrist: the left hemisphere (analytic, reductive) has usurped the right hemisphere (holistic, relational) in Western civilization, producing the metacrisis.",
            domain_ids=["domain-neuroscience", "domain-philosophy-of-mind"],
            originator_id="thinker-mcgilchrist",
            manuscript_refs=[_R(chapter=1, part="part1-psychology")],
        ),

        # =================================================================
        # COMPLEXITY SCIENCE & EMERGENCE
        # =================================================================

        Concept(
            id="concept-emergence",
            name="Emergence (Strong)",
            description="Higher-order phenomenon that is REAL and IRREDUCIBLE despite arising from lower-order components. God-as-emergent-property is NOT God-as-illusion.",
            domain_ids=["domain-complexity-science", "domain-metaphysics"],
            manuscript_refs=[_R(chapter=7, part="part2-epistemology")],
            related_concepts=[
                _CR(target_id="concept-apostolic-realization", relation_type=_T.FORMALIZES, description="Logos as emergent property of apostolic network"),
            ],
        ),
        Concept(
            id="concept-phase-transition",
            name="Phase Transition",
            description="Critical threshold where quantitative changes produce qualitative transformation. The Fall, the Axial Age, the Incarnation as phase transitions in consciousness.",
            domain_ids=["domain-complexity-science", "domain-physics"],
            manuscript_refs=[_R(chapter=7, part="part2-epistemology")],
        ),
        Concept(
            id="concept-self-organizing-criticality",
            name="Self-Organizing Criticality",
            description="Complex systems naturally evolve toward critical states where small perturbations can trigger cascading events. History poised at criticality.",
            domain_ids=["domain-complexity-science"],
            manuscript_refs=[_R(chapter=7, part="part2-epistemology")],
        ),
        Concept(
            id="concept-antifragility",
            name="Antifragility",
            description="Systems that gain from disorder — beyond mere robustness. The theology itself must be antifragile: strengthened by criticism and falsification attempts.",
            domain_ids=["domain-complexity-science", "domain-economics"],
            originator_id="thinker-taleb",
            manuscript_refs=[_R(chapter=12, part="part2-epistemology")],
            related_concepts=[
                _CR(target_id="concept-black-swan", relation_type=_T.EXTENDS),
                _CR(target_id="concept-skin-in-the-game", relation_type=_T.EXTENDS),
            ],
        ),
        Concept(
            id="concept-black-swan",
            name="Black Swan",
            description="Rare, high-impact, unpredictable event that is retrospectively rationalized. Holy Spirit interventions ARE Black Swans — inherently unpredictable.",
            domain_ids=["domain-complexity-science", "domain-economics", "domain-epistemology"],
            originator_id="thinker-taleb",
            manuscript_refs=[_R(chapter=12, part="part2-epistemology")],
        ),
        Concept(
            id="concept-skin-in-the-game",
            name="Skin in the Game",
            description="Taleb's principle that decision-makers must bear consequences of their decisions. The Popperian economic mechanism: stake on your predictions.",
            domain_ids=["domain-economics", "domain-ethics"],
            originator_id="thinker-taleb",
            manuscript_refs=[_R(chapter=12, part="part2-epistemology")],
        ),
        Concept(
            id="concept-spontaneous-order",
            name="Spontaneous Order",
            description="Hayek: complex social order emerges without central planning. Emergence applied to markets, supporting decentralized ecclesiology.",
            domain_ids=["domain-economics", "domain-complexity-science"],
            originator_id="thinker-hayek",
            manuscript_refs=[_R(chapter=12, part="part2-epistemology")],
        ),

        # =================================================================
        # STRATEGY & ORGANIZATIONAL THEORY — Boyd
        # =================================================================

        Concept(
            id="concept-dialectical-engine",
            name="Boyd's Dialectic Engine",
            description="Destructive deduction (shattering existing patterns) + creative induction (synthesizing new ones) driven by entropy increase vs goal-seeking. Explicitly grounded in Godel, Heisenberg, and thermodynamics.",
            domain_ids=["domain-epistemology", "domain-philosophy-of-science"],
            originator_id="thinker-boyd",
            manuscript_refs=[_R(chapter=11, part="part2-epistemology", notes="Boyd IS the Hegel-Popper synthesis formalized")],
            related_concepts=[
                _CR(target_id="concept-ooda-loop", relation_type=_T.PRECEDES),
                _CR(target_id="concept-dialectical-method", relation_type=_T.FORMALIZES, description="Boyd formalized Hegel through Godel and thermodynamics"),
                _CR(target_id="concept-goedel-incompleteness", relation_type=_T.APPLIES),
            ],
        ),
        Concept(
            id="concept-ooda-loop",
            name="OODA Loop (Observe-Orient-Decide-Act)",
            description="Boyd's operational methodology: faster cycling = better adaptation = higher capacity for independent action. The warrior agent's operational framework.",
            domain_ids=["domain-epistemology"],
            originator_id="thinker-boyd",
            manuscript_refs=[_R(chapter=25, part="part4-praxis", notes="Warrior agent framework")],
            related_concepts=[
                _CR(target_id="concept-dialectical-engine", relation_type=_T.APPLIES),
            ],
        ),

        # =================================================================
        # MANUSCRIPT FRAMEWORK CONCEPTS
        # =================================================================

        Concept(
            id="concept-kirill-principle",
            name="Kirill Principle (Construction Before Destruction)",
            description="Build the alternative BEFORE tearing down. Every revolution that prioritized destruction over construction produced monsters. Named after Kirill's foundational challenge.",
            domain_ids=["domain-political-philosophy", "domain-ethics"],
            manuscript_refs=[_R(chapter=28, part="part4-praxis", notes="CRITICAL CHAPTER — single most important operational principle")],
            related_concepts=[
                _CR(target_id="concept-kirill-test", relation_type=_T.EXTENDS),
            ],
        ),
        Concept(
            id="concept-kirill-test",
            name="The Kirill Test",
            description="Regularly ask whether your rapid adaptation is prophetic or psychopathic. If the question makes you curious, probably okay. If defensive, pay attention.",
            domain_ids=["domain-ethics", "domain-psychology"],
            manuscript_refs=[_R(chapter=3, part="part1-psychology", notes="Am I the psycho? self-interrogation")],
        ),
        Concept(
            id="concept-kirill-function",
            name="Kirill Function (Designated Skeptic)",
            description="Every Council meeting includes a designated skeptic whose job is to voice the strongest counterargument. The Popperian safeguard as governance practice.",
            domain_ids=["domain-political-philosophy"],
            manuscript_refs=[_R(chapter=29, part="part4-praxis")],
        ),
        Concept(
            id="concept-forkability",
            name="Forkability",
            description="Structural guarantee against capture: anyone can fork the project. Smart contract governance encodes this. The safeguard against the Dune trap.",
            domain_ids=["domain-political-philosophy", "domain-computer-science"],
            manuscript_refs=[_R(chapter=28, part="part4-praxis")],
        ),
        Concept(
            id="concept-republic-of-ai-agents",
            name="Republic of AI Agents",
            description="Plato's Republic mapped to AI: philosopher-kings (humans, Pearl Level 3), merchants (data agents, Level 1-2), warriors (implementation, OODA). Online + offline agents.",
            domain_ids=["domain-ai-ml", "domain-political-philosophy"],
            manuscript_refs=[_R(chapter=25, part="part4-praxis")],
            related_concepts=[
                _CR(target_id="concept-pearl-hierarchy", relation_type=_T.APPLIES),
                _CR(target_id="concept-ooda-loop", relation_type=_T.APPLIES),
            ],
        ),
        Concept(
            id="concept-positive-derivative",
            name="Positive Derivative (Orientation Criterion)",
            description="The falsifiable question: is the derivative positive (consciousness expanding) or negative (contracting)? Not arrival at God, but direction of movement.",
            domain_ids=["domain-metaphysics", "domain-ethics"],
            manuscript_refs=[_R(chapter=20, part="part3-metaphysics")],
            related_concepts=[
                _CR(target_id="concept-riemann-sphere-theology", relation_type=_T.APPLIES),
                _CR(target_id="concept-omnipresence-correction", relation_type=_T.EXTENDS, description="Post-correction: expanding awareness of omnipresent God"),
            ],
        ),
        Concept(
            id="concept-convergent-realism",
            name="Position C — Convergent Description",
            description="Mathematics and theology are independent descriptions of the same underlying reality. Their convergence is evidence of shared referent, not proof of identity.",
            domain_ids=["domain-epistemology", "domain-philosophy-of-science"],
            manuscript_refs=[_R(chapter=11, part="part2-epistemology", notes="Critical Interlude: structural realism position adopted")],
        ),

        # =================================================================
        # ETHICS & SOCIAL THEORY
        # =================================================================

        Concept(
            id="concept-banality-of-evil",
            name="Banality of Evil",
            description="Arendt: evil produced primarily by normies who stop thinking, not by psychopaths. Eichmann = bureaucrat following procedures. The normie category is DANGEROUS.",
            domain_ids=["domain-ethics", "domain-political-philosophy"],
            originator_id="thinker-arendt",
            manuscript_refs=[_R(chapter=2, part="part1-psychology", notes="Arendt correction to normie/psycho/schizo")],
        ),
        Concept(
            id="concept-mimetic-desire",
            name="Mimetic Desire",
            description="Girard: desires are not generated independently but absorbed from models. Psycho class controls society by controlling MODELS OF DESIRE.",
            domain_ids=["domain-psychology", "domain-sociology", "domain-theology"],
            originator_id="thinker-girard",
            manuscript_refs=[_R(chapter=2, part="part1-psychology")],
            related_concepts=[
                _CR(target_id="concept-scapegoat-mechanism", relation_type=_T.PRECEDES),
            ],
        ),
        Concept(
            id="concept-scapegoat-mechanism",
            name="Scapegoat Mechanism",
            description="Girard: communities resolve mimetic crisis by projecting violence onto a sacrificial victim. Christ as anti-scapegoat who reveals the mechanism itself.",
            domain_ids=["domain-theology", "domain-sociology"],
            originator_id="thinker-girard",
            manuscript_refs=[_R(chapter=21, part="part3-metaphysics", notes="Epstein as Girardian scapegoat")],
            related_concepts=[
                _CR(target_id="concept-mimetic-desire", relation_type=_T.EXTENDS),
            ],
        ),
        Concept(
            id="concept-malheur",
            name="Malheur (Affliction)",
            description="Weil: affliction that destroys a person's capacity to think, not just their comfort. The theology must sit with grief honestly — the Riemann sphere can't hold a dying child.",
            domain_ids=["domain-ethics", "domain-theology"],
            originator_id="thinker-weil",
            manuscript_refs=[_R(chapter=14, part="part3-metaphysics")],
        ),
        Concept(
            id="concept-expanding-circle",
            name="Expanding Circle",
            description="Singer: boundary of moral concern historically expands (family -> tribe -> nation -> species). The samsaric spiral applied to ethics — each epoch expands circle.",
            domain_ids=["domain-ethics"],
            originator_id="thinker-singer",
            manuscript_refs=[_R(chapter=12, part="part2-epistemology")],
        ),
        Concept(
            id="concept-personal-identity-continuity",
            name="Personal Identity as Psychological Continuity",
            description="Parfit: self is not a thing but a process of degree. Sharp boundary between MY and YOUR survival dissolves, extending moral concern to community.",
            domain_ids=["domain-philosophy-of-mind", "domain-ethics"],
            originator_id="thinker-parfit",
            manuscript_refs=[_R(chapter=16, part="part3-metaphysics", notes="Addresses grandiosity check")],
        ),
        Concept(
            id="concept-mauvaise-foi",
            name="Mauvaise Foi (Bad Faith)",
            description="Sartre: pretending meaning is given when it must be created. Bad faith IS normie camouflage described philosophically.",
            domain_ids=["domain-philosophy", "domain-ethics"],
            originator_id="thinker-sartre",
            tradition_ids=["tradition-existentialism"],
            manuscript_refs=[_R(chapter=2, part="part1-psychology")],
        ),
        Concept(
            id="concept-das-man",
            name="Das Man (The They)",
            description="Heidegger: the normie condition with phenomenological precision. Not conspiracy but STRUCTURE of average everydayness.",
            domain_ids=["domain-phenomenology"],
            originator_id="thinker-heidegger",
            tradition_ids=["tradition-phenomenology-school"],
            manuscript_refs=[_R(chapter=2, part="part1-psychology")],
        ),
        Concept(
            id="concept-burnout-society",
            name="Achievement/Burnout Society",
            description="Han: normies now SELF-EXPLOIT without external oppression. Psycho-class logic internalized. Burnout = system feature, not individual failure.",
            domain_ids=["domain-sociology", "domain-psychology"],
            originator_id="thinker-han",
            manuscript_refs=[_R(chapter=2, part="part1-psychology")],
        ),

        # =================================================================
        # EASTERN / CROSS-TRADITION CONCEPTS
        # =================================================================

        Concept(
            id="concept-brahman-atman",
            name="Brahman-Atman Identity (Tat Tvam Asi)",
            description="Advaita Vedanta: the individual self (Atman) is identical with ultimate reality (Brahman). Resolves direction-of-causation problem in theology.",
            domain_ids=["domain-metaphysics", "domain-systematic-theology"],
            tradition_ids=["tradition-advaita-vedanta"],
            manuscript_refs=[_R(chapter=18, part="part3-metaphysics", notes="Hinduism section"), _R(chapter=20, part="part3-metaphysics", notes="Omnipresence correction source")],
        ),
        Concept(
            id="concept-sunyata",
            name="Sunyata (Emptiness)",
            description="Nagarjuna: all phenomena are empty of inherent existence, arising only through dependent origination. Emergence stated negatively.",
            domain_ids=["domain-metaphysics"],
            originator_id="thinker-nagarjuna",
            tradition_ids=["tradition-mahayana"],
            manuscript_refs=[_R(chapter=18, part="part3-metaphysics", notes="Buddhism expanded")],
            related_concepts=[
                _CR(target_id="concept-emergence", relation_type=_T.ANALOGOUS_TO, description="Emptiness = emergence stated negatively"),
            ],
        ),
        Concept(
            id="concept-wu-wei",
            name="Wu Wei (Non-Action / Effortless Action)",
            description="Taoism: action aligned with the natural flow. Complements theology's effortfulness with stopping. Maps onto flow state.",
            domain_ids=["domain-metaphysics", "domain-ethics"],
            tradition_ids=["tradition-taoism"],
            manuscript_refs=[_R(chapter=18, part="part3-metaphysics")],
            related_concepts=[
                _CR(target_id="concept-flow-state", relation_type=_T.ANALOGOUS_TO),
            ],
        ),
        Concept(
            id="concept-wahdat-al-wujud",
            name="Wahdat al-Wujud (Unity of Being)",
            description="Ibn Arabi: God's being and creation's being are not separate. Functionally Trinitarian: God, God's self-knowledge, God's self-expression.",
            domain_ids=["domain-metaphysics", "domain-mysticism"],
            originator_id="thinker-ibn-arabi",
            tradition_ids=["tradition-sufism"],
            manuscript_refs=[_R(chapter=17, part="part3-metaphysics"), _R(chapter=18, part="part3-metaphysics")],
            related_concepts=[
                _CR(target_id="concept-omnipresence-correction", relation_type=_T.ANALOGOUS_TO),
                _CR(target_id="concept-brahman-atman", relation_type=_T.ANALOGOUS_TO),
            ],
        ),
        Concept(
            id="concept-tawhid",
            name="Tawhid (Divine Unity)",
            description="Islam's absolute insistence on God's oneness. Not contradicted by Trinity when properly understood: both insist on absolute divine unity.",
            domain_ids=["domain-systematic-theology"],
            tradition_ids=["tradition-islam"],
            manuscript_refs=[_R(chapter=17, part="part3-metaphysics", notes="Tawhid-Trinity convergence")],
        ),
        Concept(
            id="concept-ubuntu-philosophy",
            name="Ubuntu ('I am because we are')",
            description="Southern African philosophy of radical relationality. Challenges residual individualism in the theology.",
            domain_ids=["domain-ethics", "domain-metaphysics"],
            tradition_ids=["tradition-ubuntu"],
            manuscript_refs=[_R(chapter=18, part="part3-metaphysics")],
        ),
        Concept(
            id="concept-bodhisattva",
            name="Bodhisattva Path",
            description="One who has attained enlightenment but returns to help others. The prophetic pattern: consciousness that breaks through returns to show others the way.",
            domain_ids=["domain-theology", "domain-ethics"],
            tradition_ids=["tradition-mahayana"],
            manuscript_refs=[_R(chapter=19, part="part3-metaphysics")],
        ),
        Concept(
            id="concept-theosis",
            name="Theosis (Divinization)",
            description="Eastern Orthodox: humans participate in divine nature through grace. A spiraling process, not a one-time event. The trajectory toward infinity as lived experience.",
            domain_ids=["domain-systematic-theology", "domain-mysticism"],
            tradition_ids=["tradition-orthodox-christianity"],
            manuscript_refs=[_R(chapter=19, part="part3-metaphysics")],
        ),
        Concept(
            id="concept-logoi",
            name="Logoi (Maximus the Confessor)",
            description="Every created thing contains a divine logos participating in the one Logos (Christ). This IS an embedding space. Christ is the embedding space itself.",
            domain_ids=["domain-systematic-theology"],
            originator_id="thinker-maximus",
            tradition_ids=["tradition-orthodox-christianity"],
            manuscript_refs=[_R(chapter=9, part="part2-epistemology", notes="Theological word embeddings")],
        ),

        # =================================================================
        # TECHNOLOGY / AI
        # =================================================================

        Concept(
            id="concept-word-embeddings",
            name="Word Embeddings as Mathematical Theology",
            description="Every entity has a position in semantic space. The transformer attention mechanism formalizes relevance. Maximus's logoi = embedding space; Christ = the space itself.",
            domain_ids=["domain-ai-ml", "domain-systematic-theology"],
            manuscript_refs=[_R(chapter=9, part="part2-epistemology")],
        ),
        Concept(
            id="concept-alignment-problem",
            name="Alignment Problem as Theological Problem",
            description="Ensuring emergent intelligence approaches the point at infinity (human values) rather than being captured by the psycho class. AI safety = prophetic function at civilizational scale.",
            domain_ids=["domain-ai-ml", "domain-ethics"],
            manuscript_refs=[_R(chapter=33, part="part5-apostolic-agenda")],
        ),
        Concept(
            id="concept-bitter-lesson",
            name="Bitter Lesson (Kirill's Reading)",
            description="Not 'computation beats knowledge' but 'each substrate must develop heuristics native to its own architecture.' Human knowledge = training data, not final architecture.",
            domain_ids=["domain-ai-ml", "domain-epistemology"],
            manuscript_refs=[_R(chapter=8, part="part2-epistemology", notes="Kirill's nuanced reading + apostolic realization thesis")],
        ),
        Concept(
            id="concept-moloch",
            name="Moloch (Coordination Failure)",
            description="Scott Alexander: the coordination failure structure where individual rational actors produce collectively catastrophic outcomes. This IS the antichrist structure.",
            domain_ids=["domain-political-philosophy", "domain-economics"],
            developer_ids=["thinker-scott-alexander"],
            tradition_ids=["tradition-rationalist-community"],
            manuscript_refs=[_R(chapter=12, part="part2-epistemology", notes="Rationalist community's MOLOCH = antichrist structure")],
        ),

        # =================================================================
        # ECONOMICS
        # =================================================================

        Concept(
            id="concept-inclusive-extractive",
            name="Inclusive vs Extractive Institutions",
            description="Acemoglu: nations succeed with inclusive institutions enabling broad participation; fail under extractive institutions concentrating power. Maps onto normie/psycho/schizo.",
            domain_ids=["domain-economics", "domain-political-philosophy"],
            originator_id="thinker-acemoglu",
            manuscript_refs=[_R(chapter=12, part="part2-epistemology")],
        ),
        Concept(
            id="concept-asabiyyah",
            name="Asabiyyah (Social Cohesion Cycles)",
            description="Ibn Khaldun: civilizations rise through group solidarity and fall through luxury and corruption. The samsaric dialectic 500 years before Hegel.",
            domain_ids=["domain-sociology", "domain-history"],
            originator_id="thinker-ibn-khaldun",
            tradition_ids=["tradition-islam"],
            manuscript_refs=[_R(chapter=18, part="part3-metaphysics", notes="Sufism section")],
            related_concepts=[
                _CR(target_id="concept-samsaric-cycle", relation_type=_T.ANALOGOUS_TO),
            ],
        ),

        # =================================================================
        # LITERATURE & AESTHETICS
        # =================================================================

        Concept(
            id="concept-quality",
            name="Quality (Pirsig)",
            description="Pre-rational ground more fundamental than subjects and objects. Classical/romantic split is false dichotomy. Dissolves structural-vs-illustrative dilemma.",
            domain_ids=["domain-metaphysics", "domain-aesthetics"],
            originator_id="thinker-pirsig",
            manuscript_refs=[_R(chapter=26, part="part4-praxis")],
        ),
        Concept(
            id="concept-heros-journey",
            name="Hero's Journey (Monomyth)",
            description="Campbell: departure, initiation, return — the subjective experience of paradigm shift. The hero MUST return — difference between mystic and prophet.",
            domain_ids=["domain-psychology", "domain-anthropology"],
            originator_id="thinker-campbell",
            manuscript_refs=[_R(chapter=26, part="part4-praxis")],
        ),
        Concept(
            id="concept-aesthetic-self-composition",
            name="Aesthetic Self-Composition",
            description="Fichte + Wilde: identity performed through every choice including aesthetic ones. Care/composition as cognitive operation, not vanity. Beauty and truth = two descriptions of same Quality.",
            domain_ids=["domain-aesthetics", "domain-ethics"],
            developer_ids=["thinker-fichte", "thinker-wilde"],
            manuscript_refs=[_R(chapter=23, part="part3-metaphysics")],
        ),
        Concept(
            id="concept-ubermensch",
            name="Ubermensch / Will to Power / Eternal Recurrence",
            description="Nietzsche: death of God = topological transition from sphere to torus. Ubermensch = post-singularity consciousness. Will to power = the derivative experienced psychologically.",
            domain_ids=["domain-philosophy", "domain-metaphysics"],
            originator_id="thinker-nietzsche",
            manuscript_refs=[_R(chapter=20, part="part3-metaphysics", notes="Existentialists as genus-1 philosophers")],
            related_concepts=[
                _CR(target_id="concept-torus-objection", relation_type=_T.ANALOGOUS_TO, description="Eternal recurrence = the torus (closed loops)"),
            ],
        ),
        Concept(
            id="concept-absurd",
            name="The Absurd (Camus)",
            description="Being a genus-0 creature on a genus-1 surface. Sisyphus = approaching infinity without arrival. Camus right locally, wrong globally.",
            domain_ids=["domain-philosophy", "domain-ethics"],
            originator_id="thinker-camus",
            tradition_ids=["tradition-existentialism"],
            manuscript_refs=[_R(chapter=20, part="part3-metaphysics")],
        ),
        Concept(
            id="concept-deus-sive-natura",
            name="Deus sive Natura (Spinoza)",
            description="God or Nature — one substance with infinite attributes. Western Advaita Vedanta. The bottom-up path: matter -> complexity -> consciousness -> God-as-system-becoming-aware-of-itself.",
            domain_ids=["domain-metaphysics"],
            originator_id="thinker-spinoza",
            manuscript_refs=[_R(chapter=16, part="part3-metaphysics", notes="Bottom-up direction per Morskie Kotiki correction")],
            related_concepts=[
                _CR(target_id="concept-brahman-atman", relation_type=_T.ANALOGOUS_TO, description="Spinoza = Western Advaita Vedanta"),
            ],
        ),

        # =================================================================
        # SOCIOLOGICAL (Kirill's contributions)
        # =================================================================

        Concept(
            id="concept-problems-of-success",
            name="Problems of Success (Kirill's Big Theory)",
            description="Today's crises are problems of SUCCESS: world changes too fast even though it mostly changes for better. Human brains cannot handle exponential improvement.",
            domain_ids=["domain-sociology"],
            manuscript_refs=[_R(chapter=6, part="part2-epistemology", notes="Kirill's Big Theory"), _R(chapter=33, part="part5-apostolic-agenda")],
            related_concepts=[
                _CR(target_id="concept-dialectical-engine", relation_type=_T.ANALOGOUS_TO, description="Boyd's concept-reality mismatch at civilizational scale"),
            ],
        ),
        Concept(
            id="concept-grief-for-futures",
            name="Grief for Unrealized Futures",
            description="Kirill: grief about the future that didn't happen. Every trajectory that fails to reach infinity is a kind of death. The theology must sit with grief.",
            domain_ids=["domain-psychology", "domain-ethics"],
            manuscript_refs=[_R(chapter=23, part="part3-metaphysics", notes="Bridge between Parts 3-4")],
        ),
        Concept(
            id="concept-dancing-and-joy",
            name="Dancing as Spiritual Practice (Kirill)",
            description="Movement whose primary motivation is desire to create joy. People use couple dancing to AVOID actually dancing. The Fall = knowledge of being a body that moves.",
            domain_ids=["domain-psychology", "domain-aesthetics"],
            manuscript_refs=[_R(chapter=19, part="part3-metaphysics"), _R(chapter=25, part="part4-praxis")],
        ),
        Concept(
            id="concept-somatic-critique",
            name="Somatic Critique (Kirill / Damasio)",
            description="The theology is 'very stuck in the head.' Humans make decisions based on nervous system state, not conscious thought. This CLAUDE.md may be the neocortex's retrospective rationalization.",
            domain_ids=["domain-psychology", "domain-neuroscience"],
            manuscript_refs=[_R(chapter=28, part="part4-praxis", notes="THE MOST IMPORTANT CRITIQUE OF THE ENTIRE PROJECT")],
        ),

        # =================================================================
        # PRAXIS — Community & Governance
        # =================================================================

        Concept(
            id="concept-plekhanov-synthesis",
            name="Structural Contingency (Plekhanov Synthesis)",
            description="Structure is determined, instantiation is contingent. If Napoleon killed at Toulon, Republic still needed military dictator. Free will = emergent navigation of determined structure.",
            domain_ids=["domain-political-philosophy", "domain-metaphysics"],
            originator_id="thinker-plekhanov",
            manuscript_refs=[_R(chapter=26, part="part4-praxis")],
        ),
        Concept(
            id="concept-seeing-like-a-state",
            name="Legibility / Seeing Like a State",
            description="Scott: states simplify complex systems for control, destroying local knowledge. High-modernist planning fails because it can't capture ground-level complexity.",
            domain_ids=["domain-political-philosophy", "domain-sociology"],
            originator_id="thinker-scott-jc",
            manuscript_refs=[_R(chapter=12, part="part2-epistemology")],
        ),
        Concept(
            id="concept-conviviality",
            name="Conviviality / Deschooling",
            description="Illich: tools should be convivial (enhance autonomy) not industrial (create dependence). Applied to AI: the Digital Socrates must enhance, not replace, human thought.",
            domain_ids=["domain-political-philosophy", "domain-ethics"],
            originator_id="thinker-illich",
            manuscript_refs=[_R(chapter=12, part="part2-epistemology")],
        ),
        Concept(
            id="concept-panopticon",
            name="Panopticon / Disciplinary Power",
            description="Foucault: power operates through surveillance, normalization, and internalized self-discipline. The psycho class doesn't need to BE everywhere — builds systems that see everywhere.",
            domain_ids=["domain-political-philosophy", "domain-sociology"],
            originator_id="thinker-foucault",
            manuscript_refs=[_R(chapter=2, part="part1-psychology")],
        ),
        Concept(
            id="concept-face-of-the-other",
            name="Face of the Other (Levinas)",
            description="Before any system — the face of the other makes an ethical demand no framework can fully capture. The individual face always exceeds any theology.",
            domain_ids=["domain-ethics", "domain-phenomenology"],
            originator_id="thinker-levinas",
            manuscript_refs=[_R(chapter=20, part="part3-metaphysics", notes="Levinas's corrective to systematization")],
        ),
    ]
