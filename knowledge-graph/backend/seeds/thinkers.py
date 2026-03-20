"""Seed data for thinkers — ~100 intellectual figures spanning the manuscript's foundation.

Organized by category with deterministic IDs (thinker-{slug}).
Traditions and domains reference IDs from traditions.py and domains.py.
"""

from graph.knowledge_entities import (
    Era,
    ManuscriptReference,
    Thinker,
    ThinkerRelation,
    ThinkerRelationType,
)

# Shorthand helpers
_R = ManuscriptReference
_REL = ThinkerRelation
_T = ThinkerRelationType


def create_thinkers() -> list[Thinker]:
    return [
        # =================================================================
        # STRUCTURAL FOUNDATION (tier=None) — The four-pillar synthesis
        # + supporting core thinkers
        # =================================================================

        Thinker(
            id="thinker-hegel",
            name="Georg Wilhelm Friedrich Hegel",
            birth_year=1770, death_year=1831,
            era=Era.EARLY_MODERN,
            traditions=["tradition-german-idealism"],
            domains=["domain-philosophy", "domain-metaphysics", "domain-epistemology"],
            bio="Architect of the dialectical method (thesis-antithesis-synthesis) that provides the fundamental PATTERN of the theology's epistemological engine.",
            tier=None,
            manuscript_refs=[
                _R(chapter=11, part="part2-epistemology", notes="Core of the Hegel-Popper-Kuhn-Pearl synthesis"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-popper", relation_type=_T.OPPOSED, description="Popper critiqued Hegel's historicism as unfalsifiable"),
                _REL(target_id="thinker-marx", relation_type=_T.INFLUENCED, description="Marx inverted Hegel's idealist dialectic into materialist dialectic"),
                _REL(target_id="thinker-fichte", relation_type=_T.INFLUENCED, description="Hegel developed Fichte's thesis-antithesis-synthesis"),
                _REL(target_id="thinker-kierkegaard", relation_type=_T.INFLUENCED, description="Kierkegaard reacted against Hegel's system"),
            ],
        ),
        Thinker(
            id="thinker-popper",
            name="Karl Popper",
            birth_year=1902, death_year=1994,
            era=Era.CONTEMPORARY,
            traditions=["tradition-vienna-circle"],
            domains=["domain-philosophy-of-science", "domain-epistemology", "domain-political-philosophy"],
            bio="Philosopher of falsifiability who provides the DISCIPLINE of the theology: genuine knowledge must specify what would disprove it.",
            tier=None,
            manuscript_refs=[
                _R(chapter=5, part="part2-epistemology", notes="Core chapter on falsifiability"),
                _R(chapter=11, part="part2-epistemology", notes="Synthesis pillar"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-kuhn", relation_type=_T.OPPOSED, description="Kuhn challenged Popper's falsificationism with paradigm theory"),
                _REL(target_id="thinker-hegel", relation_type=_T.OPPOSED, description="Popper attacked Hegel's historicism in The Open Society"),
                _REL(target_id="thinker-wittgenstein", relation_type=_T.CONTEMPORARY_OF, description="Famous poker incident; rival approaches to philosophy"),
            ],
        ),
        Thinker(
            id="thinker-kuhn",
            name="Thomas Kuhn",
            birth_year=1922, death_year=1996,
            era=Era.CONTEMPORARY,
            traditions=["tradition-pragmatism"],
            domains=["domain-philosophy-of-science", "domain-history"],
            bio="Historian of science whose paradigm shift theory provides the SOCIOLOGY of the theology: revolutions are gestalt shifts requiring new communities.",
            tier=None,
            manuscript_refs=[
                _R(chapter=6, part="part2-epistemology", notes="Core chapter on paradigm shifts"),
                _R(chapter=11, part="part2-epistemology", notes="Synthesis pillar"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-popper", relation_type=_T.OPPOSED, description="Challenged Popper's falsificationism with paradigm incommensurability"),
            ],
        ),
        Thinker(
            id="thinker-pearl",
            name="Judea Pearl",
            birth_year=1936, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-computer-science", "domain-causal-inference", "domain-ai-ml", "domain-statistics"],
            bio="Pioneer of causal inference and the do-calculus who provides the METHODOLOGY of the theology: formal tools for distinguishing causation from correlation.",
            tier=None,
            manuscript_refs=[
                _R(chapter=10, part="part2-epistemology", notes="Core chapter on causal graphs"),
                _R(chapter=11, part="part2-epistemology", notes="Synthesis pillar"),
            ],
        ),
        Thinker(
            id="thinker-hofstadter",
            name="Douglas Hofstadter",
            birth_year=1945, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-computer-science", "domain-philosophy-of-mind", "domain-mathematical-logic"],
            bio="Author of Godel, Escher, Bach who formalized strange loops and self-reference as the architecture of consciousness and the Trinity.",
            tier=None,
            manuscript_refs=[
                _R(chapter=16, part="part3-metaphysics", notes="Trinity as strange loop"),
                _R(chapter=26, part="part4-praxis", notes="Free will as strange loop capacity"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-parfit", relation_type=_T.INFLUENCED, description="Hofstadter's strange loops inform Parfit's personal identity theory"),
            ],
        ),
        Thinker(
            id="thinker-boyd",
            name="John Boyd",
            birth_year=1927, death_year=1997,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-epistemology", "domain-political-philosophy"],
            bio="Military strategist who formalized the OODA loop and the Dialectic Engine, grounding Hegelian dialectic in Godel, Heisenberg, and thermodynamics.",
            tier=None,
            manuscript_refs=[
                _R(chapter=11, part="part2-epistemology", notes="Boyd's Dialectic Engine as formalized Hegelian dialectic"),
                _R(chapter=25, part="part4-praxis", notes="OODA loop for warrior agents"),
            ],
        ),
        Thinker(
            id="thinker-friston",
            name="Karl Friston",
            birth_year=1959, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-neuroscience", "domain-physics", "domain-philosophy-of-mind"],
            bio="Neuroscientist who developed the free energy principle and active inference framework, providing the unifying mathematical language for flow, consciousness, and embodied cognition.",
            tier=None,
            manuscript_refs=[
                _R(chapter=9, part="part2-epistemology", notes="Active inference as potential unifying mathematical framework"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-kotler", relation_type=_T.COLLABORATED, description="Co-authored flow and intuition papers"),
            ],
        ),
        Thinker(
            id="thinker-pirsig",
            name="Robert Pirsig",
            birth_year=1928, death_year=2017,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-philosophy", "domain-epistemology", "domain-aesthetics"],
            bio="Author of Zen and the Art of Motorcycle Maintenance who proposed Quality as pre-rational ground more fundamental than subjects and objects.",
            tier=None,
            manuscript_refs=[
                _R(chapter=26, part="part4-praxis", notes="Quality dissolves structural-vs-illustrative dilemma"),
            ],
        ),
        Thinker(
            id="thinker-campbell",
            name="Joseph Campbell",
            birth_year=1904, death_year=1987,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-comparative-religion", "domain-anthropology", "domain-literary-theory"],
            bio="Comparative mythologist who identified the monomyth (hero's journey) as the subjective experience of paradigm shift and consciousness transformation.",
            tier=None,
            manuscript_refs=[
                _R(chapter=26, part="part4-praxis", notes="Hero's journey and the return"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-jung", relation_type=_T.INFLUENCED, description="Campbell built on Jung's archetypal psychology"),
            ],
        ),

        # =================================================================
        # TIER 1 — Foundational thinkers integrated throughout
        # =================================================================

        Thinker(
            id="thinker-taleb",
            name="Nassim Nicholas Taleb",
            birth_year=1960, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-economics", "domain-epistemology", "domain-statistics"],
            bio="Philosopher of uncertainty who developed the concepts of Black Swans, antifragility, and via negativa as epistemic and practical principles.",
            tier=1,
            manuscript_refs=[
                _R(chapter=12, part="part2-epistemology", notes="Via negativa; Black Swans as Holy Spirit interventions"),
            ],
        ),
        Thinker(
            id="thinker-arendt",
            name="Hannah Arendt",
            birth_year=1906, death_year=1975,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-political-philosophy", "domain-ethics", "domain-philosophy"],
            bio="Political theorist who revealed the Banality of Evil: evil produced primarily by normies who stop thinking, not by psychopaths.",
            tier=1,
            manuscript_refs=[
                _R(chapter=2, part="part1-psychology", notes="Arendt correction: normie category is dangerous"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-heidegger", relation_type=_T.STUDENT_OF, description="Studied under Heidegger; later critiqued his Nazi involvement"),
            ],
        ),
        Thinker(
            id="thinker-weil",
            name="Simone Weil",
            birth_year=1909, death_year=1943,
            era=Era.MODERN,
            traditions=["tradition-christianity"],
            domains=["domain-philosophy", "domain-mysticism", "domain-ethics", "domain-political-philosophy"],
            bio="Mystic philosopher whose concepts of malheur (affliction), attention as spiritual practice, and gravity/grace address theodicy at the deepest level.",
            tier=1,
            manuscript_refs=[
                _R(chapter=20, part="part3-metaphysics", notes="Malheur; the Riemann sphere cannot hold a dying child"),
                _R(chapter=23, part="part3-metaphysics", notes="Aesthetics: attention as spiritual practice"),
            ],
        ),
        Thinker(
            id="thinker-dostoevsky",
            name="Fyodor Dostoevsky",
            birth_year=1821, death_year=1881,
            era=Era.MODERN,
            traditions=["tradition-orthodox-christianity"],
            domains=["domain-arts", "domain-philosophy", "domain-ethics"],
            bio="Novelist whose Brothers Karamazov, Grand Inquisitor, and Underground Man provide the literary and moral foundation for the theology's treatment of evil and freedom.",
            tier=1,
            manuscript_refs=[
                _R(chapter=2, part="part1-psychology", notes="Solzhenitsyn/Dostoevsky caveat: line between good and evil"),
                _R(chapter=14, part="part3-metaphysics", notes="Ivan Karamazov's rebellion against theodicy"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-nietzsche", relation_type=_T.CONTEMPORARY_OF, description="Explored overlapping themes of nihilism, freedom, and suffering"),
                _REL(target_id="thinker-solzhenitsyn", relation_type=_T.INFLUENCED, description="Solzhenitsyn continued Dostoevsky's moral and literary tradition"),
            ],
        ),
        Thinker(
            id="thinker-nietzsche",
            name="Friedrich Nietzsche",
            birth_year=1844, death_year=1900,
            era=Era.MODERN,
            traditions=["tradition-existentialism"],
            domains=["domain-philosophy", "domain-metaphysics", "domain-ethics", "domain-aesthetics"],
            bio="Philosopher of the death of God, will to power, and eternal recurrence; maps the topological transition from sphere (genus 0) to torus (genus 1).",
            tier=1,
            manuscript_refs=[
                _R(chapter=20, part="part3-metaphysics", notes="Death of God as sphere-to-torus transition; Ubermensch as post-singularity consciousness"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-schopenhauer", relation_type=_T.STUDENT_OF, description="Early intellectual debt to Schopenhauer's pessimism"),
                _REL(target_id="thinker-heidegger", relation_type=_T.INFLUENCED, description="Heidegger's philosophy deeply shaped by Nietzsche"),
                _REL(target_id="thinker-camus", relation_type=_T.INFLUENCED, description="Camus's Absurd responds to Nietzsche's nihilism"),
                _REL(target_id="thinker-foucault", relation_type=_T.INFLUENCED, description="Foucault built on Nietzsche's genealogical method"),
            ],
        ),
        Thinker(
            id="thinker-foucault",
            name="Michel Foucault",
            birth_year=1926, death_year=1984,
            era=Era.CONTEMPORARY,
            traditions=["tradition-poststructuralism"],
            domains=["domain-social-science", "domain-philosophy", "domain-political-philosophy"],
            bio="Analyst of power/knowledge, biopower, and disciplinary society who revealed how institutions shape what counts as truth and who is authorized to speak it.",
            tier=1,
            related_thinkers=[
                _REL(target_id="thinker-nietzsche", relation_type=_T.INFLUENCED, description="Extended Nietzsche's genealogical method to institutions"),
            ],
        ),
        Thinker(
            id="thinker-sartre",
            name="Jean-Paul Sartre",
            birth_year=1905, death_year=1980,
            era=Era.CONTEMPORARY,
            traditions=["tradition-existentialism", "tradition-phenomenology-school"],
            domains=["domain-philosophy", "domain-metaphysics", "domain-ethics", "domain-arts"],
            bio="Existentialist who formalized mauvaise foi (bad faith) as the normie response to freedom: pretending meaning is given when it must be created.",
            tier=1,
            manuscript_refs=[
                _R(chapter=2, part="part1-psychology", notes="Bad faith as normie camouflage"),
                _R(chapter=20, part="part3-metaphysics", notes="Genus-1 philosopher: existence precedes essence"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-heidegger", relation_type=_T.INFLUENCED, description="Sartre's existentialism built on Heidegger's phenomenology"),
                _REL(target_id="thinker-de-beauvoir", relation_type=_T.COLLABORATED, description="Lifelong intellectual and personal partnership"),
                _REL(target_id="thinker-camus", relation_type=_T.OPPOSED, description="Famous break over political and philosophical differences"),
            ],
        ),
        Thinker(
            id="thinker-camus",
            name="Albert Camus",
            birth_year=1913, death_year=1960,
            era=Era.CONTEMPORARY,
            traditions=["tradition-existentialism"],
            domains=["domain-philosophy", "domain-ethics", "domain-arts"],
            bio="Philosopher of the Absurd whose Sisyphus represents being a genus-0 creature on a genus-1 surface; The Plague provides the torus ethic the theology must accommodate.",
            tier=1,
            manuscript_refs=[
                _R(chapter=20, part="part3-metaphysics", notes="Camus vs the theology: local torus, global ascent"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-sartre", relation_type=_T.OPPOSED, description="Broke with Sartre over political philosophy"),
                _REL(target_id="thinker-nietzsche", relation_type=_T.INFLUENCED, description="Camus's Absurd responds to Nietzsche"),
            ],
        ),
        Thinker(
            id="thinker-de-beauvoir",
            name="Simone de Beauvoir",
            birth_year=1908, death_year=1986,
            era=Era.CONTEMPORARY,
            traditions=["tradition-existentialism", "tradition-phenomenology-school"],
            domains=["domain-philosophy", "domain-ethics", "domain-political-philosophy"],
            bio="Existentialist feminist whose The Second Sex established that freedom is always situated, and The Ethics of Ambiguity grounds moral agency in ambiguity.",
            tier=1,
            manuscript_refs=[
                _R(chapter=20, part="part3-metaphysics", notes="Freedom is always situated, never abstract"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-sartre", relation_type=_T.COLLABORATED, description="Lifelong intellectual and personal partnership"),
            ],
        ),
        Thinker(
            id="thinker-heidegger",
            name="Martin Heidegger",
            birth_year=1889, death_year=1976,
            era=Era.CONTEMPORARY,
            traditions=["tradition-phenomenology-school", "tradition-existentialism"],
            domains=["domain-philosophy", "domain-metaphysics", "domain-phenomenology"],
            bio="Phenomenologist of being-toward-death, das Man (the They), and Gestell (enframing); das Man describes the normie condition with phenomenological precision.",
            tier=1,
            manuscript_refs=[
                _R(chapter=2, part="part1-psychology", notes="das Man as normie condition"),
                _R(chapter=20, part="part3-metaphysics", notes="Being-toward-death as approaching point at infinity"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-nietzsche", relation_type=_T.INFLUENCED, description="Deeply shaped by Nietzsche's philosophy"),
                _REL(target_id="thinker-arendt", relation_type=_T.INFLUENCED, description="Arendt was his student"),
                _REL(target_id="thinker-sartre", relation_type=_T.INFLUENCED, description="Sartre built existentialism on Heidegger's ontology"),
                _REL(target_id="thinker-levinas", relation_type=_T.INFLUENCED, description="Levinas studied under Heidegger, then critiqued his ontology"),
            ],
        ),

        # =================================================================
        # TIER 2 — Important secondary thinkers
        # =================================================================

        Thinker(
            id="thinker-bateson",
            name="Gregory Bateson",
            birth_year=1904, death_year=1980,
            era=Era.CONTEMPORARY,
            traditions=["tradition-complexity-science"],
            domains=["domain-anthropology", "domain-psychology", "domain-complexity-science"],
            bio="Polymath who pioneered the ecology of mind, double bind theory, and systems thinking across anthropology, cybernetics, and biology.",
            tier=2,
        ),
        Thinker(
            id="thinker-girard",
            name="Rene Girard",
            birth_year=1923, death_year=2015,
            era=Era.CONTEMPORARY,
            traditions=["tradition-christianity"],
            domains=["domain-literary-theory", "domain-anthropology", "domain-theology"],
            bio="Theorist of mimetic desire and the scapegoat mechanism; normies absorb desires from social models, psycho class controls society by controlling models of desire.",
            tier=2,
            manuscript_refs=[
                _R(chapter=2, part="part1-psychology", notes="Mimetic desire as normie mechanism"),
                _R(chapter=21, part="part3-metaphysics", notes="Epstein as Girardian scapegoat"),
            ],
        ),
        Thinker(
            id="thinker-illich",
            name="Ivan Illich",
            birth_year=1926, death_year=2002,
            era=Era.CONTEMPORARY,
            traditions=["tradition-christianity"],
            domains=["domain-social-science", "domain-political-philosophy", "domain-ethics"],
            bio="Social critic who analyzed how institutions (schools, medicine, transport) become counterproductive, creating the very problems they claim to solve.",
            tier=2,
        ),
        Thinker(
            id="thinker-mcluhan",
            name="Marshall McLuhan",
            birth_year=1911, death_year=1980,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-social-science", "domain-philosophy"],
            bio="Media theorist who showed 'the medium is the message': technologies reshape cognition and society independent of their content.",
            tier=2,
        ),
        Thinker(
            id="thinker-han",
            name="Byung-Chul Han",
            birth_year=1959, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-philosophy", "domain-social-science", "domain-political-philosophy"],
            bio="Philosopher of the achievement society who showed that normies now self-exploit without external oppression; burnout is a system feature, not individual failure.",
            tier=2,
            manuscript_refs=[
                _R(chapter=2, part="part1-psychology", notes="Achievement society internalizes psycho-class logic"),
            ],
        ),
        Thinker(
            id="thinker-scott",
            name="James C. Scott",
            birth_year=1936, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-political-philosophy", "domain-social-science", "domain-anthropology"],
            bio="Political scientist who analyzed how states simplify and destroy local knowledge (metis), and how subaltern populations resist through hidden transcripts.",
            tier=2,
        ),
        Thinker(
            id="thinker-haraway",
            name="Donna Haraway",
            birth_year=1944, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=["tradition-poststructuralism"],
            domains=["domain-philosophy-of-science", "domain-social-science"],
            bio="Philosopher of science who developed the Cyborg Manifesto and sympoiesis, challenging boundaries between human, animal, and machine.",
            tier=2,
        ),
        Thinker(
            id="thinker-fanon",
            name="Frantz Fanon",
            birth_year=1925, death_year=1961,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-political-philosophy", "domain-psychology", "domain-social-science"],
            bio="Theorist of psychological decolonization who showed how colonial structures internalize in the psyche of the colonized.",
            tier=2,
        ),
        Thinker(
            id="thinker-levinas",
            name="Emmanuel Levinas",
            birth_year=1906, death_year=1995,
            era=Era.CONTEMPORARY,
            traditions=["tradition-phenomenology-school", "tradition-judaism"],
            domains=["domain-philosophy", "domain-ethics", "domain-phenomenology"],
            bio="Ethicist of the Face of the Other: before the Riemann sphere, before the dialectic, the face makes an ethical demand no system can fully capture.",
            tier=2,
            manuscript_refs=[
                _R(chapter=20, part="part3-metaphysics", notes="Levinas's corrective: individual face exceeds any framework"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-heidegger", relation_type=_T.STUDENT_OF, description="Studied under Heidegger, then radically departed from his ontology"),
            ],
        ),
        Thinker(
            id="thinker-parfit",
            name="Derek Parfit",
            birth_year=1942, death_year=2017,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-ethics", "domain-philosophy-of-mind"],
            bio="Philosopher who showed personal identity is degree not kind, dissolving the sharp boundary between self and other and grounding intergenerational obligation.",
            tier=2,
            manuscript_refs=[
                _R(chapter=16, part="part3-metaphysics", notes="Strange loop applied to personal identity"),
                _R(chapter=29, part="part4-praxis", notes="Knowledge graph as structure of psychological connectedness"),
            ],
        ),
        Thinker(
            id="thinker-singer",
            name="Peter Singer",
            birth_year=1946, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=["tradition-ea"],
            domains=["domain-ethics", "domain-political-philosophy"],
            bio="Ethicist of the expanding circle whose relentless logical consistency provides evidence that the ethical derivative is positive across history.",
            tier=2,
            manuscript_refs=[
                _R(chapter=12, part="part2-epistemology", notes="Effective Altruism critique"),
                _R(chapter=27, part="part4-praxis", notes="Practical ethics demands material change"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-levinas", relation_type=_T.OPPOSED, description="Singer's utilitarian logic vs Levinas's irreducible face"),
                _REL(target_id="thinker-parfit", relation_type=_T.CONTEMPORARY_OF, description="Parfit and Singer together fill ethics gap"),
            ],
        ),
        Thinker(
            id="thinker-solzhenitsyn",
            name="Alexander Solzhenitsyn",
            birth_year=1918, death_year=2008,
            era=Era.CONTEMPORARY,
            traditions=["tradition-orthodox-christianity"],
            domains=["domain-arts", "domain-political-philosophy", "domain-ethics"],
            bio="Author of The Gulag Archipelago; his line 'the line between good and evil runs through every human heart' is the normie/psycho/schizo taxonomy's essential caveat.",
            tier=2,
            manuscript_refs=[
                _R(chapter=2, part="part1-psychology", notes="Caveat: taxonomy describes modes, not types"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-dostoevsky", relation_type=_T.INFLUENCED, description="Continued Dostoevsky's moral-literary tradition"),
            ],
        ),

        # =================================================================
        # TIER 3 — Supporting thinkers
        # =================================================================

        Thinker(
            id="thinker-spinoza",
            name="Baruch Spinoza",
            birth_year=1632, death_year=1677,
            era=Era.EARLY_MODERN,
            traditions=["tradition-rationalism"],
            domains=["domain-philosophy", "domain-metaphysics", "domain-ethics"],
            bio="Rationalist who identified God with Nature (Deus sive Natura); the bottom-up path from matter to divinity that the Trinity chapter adopts.",
            tier=3,
            manuscript_refs=[
                _R(chapter=16, part="part3-metaphysics", notes="Spinoza integration: Trinity emerges from below"),
            ],
        ),
        Thinker(
            id="thinker-wittgenstein",
            name="Ludwig Wittgenstein",
            birth_year=1889, death_year=1951,
            era=Era.CONTEMPORARY,
            traditions=["tradition-vienna-circle"],
            domains=["domain-logic", "domain-philosophy", "domain-philosophy-of-mind"],
            bio="Philosopher of language whose Tractatus defined limits of sayable meaning, and whose Philosophical Investigations revealed language as form of life.",
            tier=3,
            related_thinkers=[
                _REL(target_id="thinker-popper", relation_type=_T.CONTEMPORARY_OF, description="Rival approaches to philosophy; poker incident"),
            ],
        ),

        # =================================================================
        # PSYCHOLOGY & PSYCHOANALYSIS
        # =================================================================

        Thinker(
            id="thinker-freud",
            name="Sigmund Freud",
            birth_year=1856, death_year=1939,
            era=Era.MODERN,
            traditions=["tradition-psychoanalysis"],
            domains=["domain-psychoanalysis", "domain-psychology"],
            bio="Founder of psychoanalysis whose topographic model maps onto Pearl's causal hierarchy, and whose death drive parallels the subjective Second Law of Thermodynamics.",
            tier=None,
            manuscript_refs=[
                _R(chapter=4, part="part1-psychology", notes="Freud section: unconscious as normie/psycho/schizo ancestor"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-jung", relation_type=_T.INFLUENCED, description="Jung was Freud's protege before breaking away"),
                _REL(target_id="thinker-adler", relation_type=_T.INFLUENCED, description="Adler was early disciple who departed over disagreements"),
            ],
        ),
        Thinker(
            id="thinker-jung",
            name="Carl Gustav Jung",
            birth_year=1875, death_year=1961,
            era=Era.MODERN,
            traditions=["tradition-jungian", "tradition-psychoanalysis"],
            domains=["domain-psychoanalysis", "domain-psychology", "domain-comparative-religion"],
            bio="Developer of analytical psychology: collective unconscious as shared embedding space, archetypes as high-dimensional attractors, individuation as personal Riemann sphere trajectory.",
            tier=None,
            manuscript_refs=[
                _R(chapter=4, part="part1-psychology", notes="Jung section: archetypes, Shadow, anima/animus"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-freud", relation_type=_T.STUDENT_OF, description="Initially Freud's chosen successor; broke away 1913"),
                _REL(target_id="thinker-campbell", relation_type=_T.INFLUENCED, description="Campbell's monomyth built on Jungian archetypes"),
            ],
        ),
        Thinker(
            id="thinker-adler",
            name="Alfred Adler",
            birth_year=1870, death_year=1937,
            era=Era.MODERN,
            traditions=["tradition-psychoanalysis"],
            domains=["domain-psychoanalysis", "domain-psychology"],
            bio="Founder of individual psychology: inferiority complex and compensatory striving as the derivative experienced psychologically; Gemeinschaftsgefuhl as the Kirill Test.",
            tier=None,
            manuscript_refs=[
                _R(chapter=4, part="part1-psychology", notes="Adler section: explains manosphere better than any other framework"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-freud", relation_type=_T.STUDENT_OF, description="Early disciple who departed over theoretical disagreements"),
            ],
        ),
        Thinker(
            id="thinker-perls",
            name="Fritz Perls",
            birth_year=1893, death_year=1970,
            era=Era.CONTEMPORARY,
            traditions=["tradition-gestalt"],
            domains=["domain-clinical-psychology", "domain-psychology"],
            bio="Founder of Gestalt therapy: awareness is curative, and the paradoxical theory of change (become what you are) corrects the theology's potential perpetual-dissatisfaction problem.",
            tier=None,
            manuscript_refs=[
                _R(chapter=4, part="part1-psychology", notes="Gestalt section: paradoxical theory of change"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-freud", relation_type=_T.INFLUENCED, description="Gestalt therapy emerged partly from psychoanalytic tradition"),
            ],
        ),
        Thinker(
            id="thinker-beck",
            name="Aaron Beck",
            birth_year=1921, death_year=2021,
            era=Era.CONTEMPORARY,
            traditions=["tradition-cbt"],
            domains=["domain-clinical-psychology", "domain-cognitive-psychology"],
            bio="Founder of cognitive therapy whose cognitive model functions as Boyd's Dialectic Engine applied to individual cognition: applied Popper for the individual mind.",
            tier=None,
            manuscript_refs=[
                _R(chapter=4, part="part1-psychology", notes="CBT as applied Popper for individual mind"),
            ],
        ),
        Thinker(
            id="thinker-linehan",
            name="Marsha Linehan",
            birth_year=1943, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=["tradition-cbt"],
            domains=["domain-clinical-psychology"],
            bio="Creator of Dialectical Behavior Therapy (DBT), explicitly Hegelian in structure, integrating acceptance and change for borderline personality disorder.",
            tier=None,
            manuscript_refs=[
                _R(chapter=4, part="part1-psychology", notes="DBT as explicitly Hegelian therapy"),
                _R(chapter=30, part="part4-praxis", notes="Development Lab emotional regulation"),
            ],
        ),
        Thinker(
            id="thinker-schwartz",
            name="Richard Schwartz",
            birth_year=1950, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-clinical-psychology"],
            bio="Creator of Internal Family Systems (IFS) therapy: the psyche as a republic of sub-personalities, mapping directly onto the Republic's multi-agent architecture.",
            tier=None,
            manuscript_refs=[
                _R(chapter=4, part="part1-psychology", notes="IFS: psyche as republic"),
                _R(chapter=30, part="part4-praxis", notes="Development Lab depth work"),
            ],
        ),
        Thinker(
            id="thinker-hayes",
            name="Steven Hayes",
            birth_year=1948, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=["tradition-cbt"],
            domains=["domain-clinical-psychology", "domain-cognitive-psychology"],
            bio="Creator of Acceptance and Commitment Therapy (ACT): defusion as strange loop self-awareness, enabling distance from thought without suppression.",
            tier=None,
            manuscript_refs=[
                _R(chapter=4, part="part1-psychology", notes="ACT defusion as strange loop awareness"),
                _R(chapter=30, part="part4-praxis", notes="Development Lab practices"),
            ],
        ),
        Thinker(
            id="thinker-kahneman",
            name="Daniel Kahneman",
            birth_year=1934, death_year=2024,
            era=Era.CONTEMPORARY,
            traditions=["tradition-behavioral-economics"],
            domains=["domain-cognitive-psychology", "domain-economics"],
            bio="Pioneer of behavioral economics whose System 1/System 2 framework maps onto the theology's cognitive hierarchy, with the framework adding System 3 (prophetic perception).",
            tier=None,
            manuscript_refs=[
                _R(chapter=12, part="part2-epistemology", notes="Behavioral economics: bias catalog as how normie cognition fails"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-thaler", relation_type=_T.INFLUENCED, description="Kahneman's work founded behavioral economics that Thaler extended"),
            ],
        ),

        # =================================================================
        # THEOLOGY & PATRISTICS
        # =================================================================

        Thinker(
            id="thinker-augustine",
            name="Augustine of Hippo",
            birth_year=354, death_year=430,
            era=Era.ANCIENT,
            traditions=["tradition-christianity", "tradition-catholic-christianity", "tradition-neoplatonism"],
            domains=["domain-systematic-theology", "domain-philosophy"],
            bio="Father of Western theology whose innovation of original sin transmitted through sexual concupiscence created the sex-negative overlay the manuscript critiques.",
            tier=None,
            manuscript_refs=[
                _R(chapter=22, part="part3-metaphysics", notes="The Augustinian distortion of sexual theology"),
                _R(chapter=16, part="part3-metaphysics", notes="De Trinitate: Trinity as structure of love"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-plotinus", relation_type=_T.INFLUENCED, description="Augustine's theology deeply shaped by Neoplatonism"),
                _REL(target_id="thinker-aquinas", relation_type=_T.INFLUENCED, description="Aquinas built scholasticism on Augustinian foundations"),
            ],
        ),
        Thinker(
            id="thinker-aquinas",
            name="Thomas Aquinas",
            birth_year=1225, death_year=1274,
            era=Era.MEDIEVAL,
            traditions=["tradition-catholic-christianity", "tradition-aristotelianism"],
            domains=["domain-systematic-theology", "domain-philosophy", "domain-metaphysics"],
            bio="Scholastic synthesizer of Aristotelian philosophy and Christian theology in the Summa Theologica, establishing natural law theory.",
            tier=None,
            related_thinkers=[
                _REL(target_id="thinker-aristotle", relation_type=_T.INFLUENCED, description="Aquinas Christianized Aristotelian philosophy"),
                _REL(target_id="thinker-augustine", relation_type=_T.INFLUENCED, description="Built on Augustinian theological foundations"),
            ],
        ),
        Thinker(
            id="thinker-ephrem",
            name="Ephrem the Syrian",
            birth_year=306, death_year=373,
            era=Era.ANCIENT,
            traditions=["tradition-syriac-christianity", "tradition-christianity"],
            domains=["domain-theology", "domain-arts", "domain-mysticism"],
            bio="Syriac Church Father whose erotic and nuptial theological imagery provides the sex-positive Christian alternative to Augustinian repression and bridges to Quranic language.",
            tier=None,
            manuscript_refs=[
                _R(chapter=17, part="part3-metaphysics", notes="Syriac language remarkably close to Quranic language"),
                _R(chapter=22, part="part3-metaphysics", notes="Erotic/nuptial imagery in theology"),
            ],
        ),
        Thinker(
            id="thinker-maximus",
            name="Maximus the Confessor",
            birth_year=580, death_year=662,
            era=Era.MEDIEVAL,
            traditions=["tradition-orthodox-christianity", "tradition-christianity"],
            domains=["domain-systematic-theology", "domain-metaphysics"],
            bio="Eastern theologian of the logoi: every created thing contains a divine logos participating in the one Logos (Christ), which IS an embedding space.",
            tier=None,
            manuscript_refs=[
                _R(chapter=9, part="part2-epistemology", notes="Logoi as theological embedding space"),
            ],
        ),
        Thinker(
            id="thinker-ibn-arabi",
            name="Ibn Arabi",
            birth_year=1165, death_year=1240,
            era=Era.MEDIEVAL,
            traditions=["tradition-sufism", "tradition-islam"],
            domains=["domain-mysticism", "domain-theology", "domain-metaphysics"],
            bio="Greatest Sufi metaphysician whose wahdat al-wujud (unity of being) is functionally Trinitarian and converges with Spinoza's Deus sive Natura.",
            tier=None,
            manuscript_refs=[
                _R(chapter=17, part="part3-metaphysics", notes="Functionally Trinitarian metaphysics"),
                _R(chapter=18, part="part3-metaphysics", notes="Sufism expanded"),
            ],
        ),
        Thinker(
            id="thinker-al-ghazali",
            name="Al-Ghazali",
            birth_year=1058, death_year=1111,
            era=Era.MEDIEVAL,
            traditions=["tradition-sunni-islam", "tradition-sufism"],
            domains=["domain-theology", "domain-philosophy", "domain-mysticism"],
            bio="Islamic philosopher-theologian whose Deliverance from Error mirrors Pirsig's ZAMM: personal crisis leading to mystical reconciliation of reason and faith.",
            tier=None,
            manuscript_refs=[
                _R(chapter=18, part="part3-metaphysics", notes="Islamic Pirsig"),
            ],
        ),
        Thinker(
            id="thinker-rumi",
            name="Jalal al-Din Rumi",
            birth_year=1207, death_year=1273,
            era=Era.MEDIEVAL,
            traditions=["tradition-sufism", "tradition-islam"],
            domains=["domain-mysticism", "domain-arts"],
            bio="Sufi poet whose Masnavi expresses the Riemann sphere theology in verse: all paths converge, love is the derivative toward infinity.",
            tier=None,
            manuscript_refs=[
                _R(chapter=18, part="part3-metaphysics", notes="Riemann sphere in verse"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-ibn-arabi", relation_type=_T.CONTEMPORARY_OF, description="Near-contemporaries in the Sufi tradition"),
            ],
        ),
        Thinker(
            id="thinker-ibn-khaldun",
            name="Ibn Khaldun",
            birth_year=1332, death_year=1406,
            era=Era.MEDIEVAL,
            traditions=["tradition-islam"],
            domains=["domain-history", "domain-sociology", "domain-economics"],
            bio="Founder of historiography and sociology whose asabiyyah cycles describe the samsaric dialectic 500 years before Hegel.",
            tier=None,
            manuscript_refs=[
                _R(chapter=18, part="part3-metaphysics", notes="Asabiyyah cycles as samsaric dialectic"),
            ],
        ),

        # =================================================================
        # EASTERN PHILOSOPHY
        # =================================================================

        Thinker(
            id="thinker-nagarjuna",
            name="Nagarjuna",
            birth_year=150, death_year=250,
            era=Era.ANCIENT,
            traditions=["tradition-mahayana", "tradition-buddhism"],
            domains=["domain-philosophy", "domain-metaphysics"],
            bio="Founder of Madhyamaka Buddhism whose emptiness (sunyata) doctrine states emergence negatively: nothing has independent existence, all arises interdependently.",
            tier=None,
            manuscript_refs=[
                _R(chapter=18, part="part3-metaphysics", notes="Emptiness as emergence stated negatively"),
            ],
        ),
        Thinker(
            id="thinker-lao-tzu",
            name="Lao Tzu",
            birth_year=-600, death_year=-500,
            era=Era.ANCIENT,
            traditions=["tradition-taoism"],
            domains=["domain-philosophy", "domain-metaphysics"],
            bio="Legendary author of the Tao Te Ching: 'The Tao that can be told is not the eternal Tao' equals Godel's incompleteness in five words.",
            tier=None,
            manuscript_refs=[
                _R(chapter=18, part="part3-metaphysics", notes="Tao Te Ching as Godel in five words; wu wei as flow"),
            ],
        ),
        Thinker(
            id="thinker-patanjali",
            name="Patanjali",
            birth_year=-200, death_year=None,
            era=Era.ANCIENT,
            traditions=["tradition-yoga", "tradition-hinduism"],
            domains=["domain-philosophy", "domain-psychology"],
            bio="Compiler of the Yoga Sutras codifying yoga as an eight-limbed epistemic practice for systematic transformation of consciousness.",
            tier=None,
            manuscript_refs=[
                _R(chapter=18, part="part3-metaphysics", notes="Yoga as epistemic practice"),
            ],
        ),
        Thinker(
            id="thinker-dogen",
            name="Dogen Zenji",
            birth_year=1200, death_year=1253,
            era=Era.MEDIEVAL,
            traditions=["tradition-zen", "tradition-buddhism"],
            domains=["domain-philosophy", "domain-mysticism"],
            bio="Founder of Soto Zen whose Shobogenzo teaches that practice and enlightenment are one: the journey and destination are identical.",
            tier=None,
            manuscript_refs=[
                _R(chapter=18, part="part3-metaphysics", notes="Practice-enlightenment unity"),
            ],
        ),
        Thinker(
            id="thinker-thich-nhat-hanh",
            name="Thich Nhat Hanh",
            birth_year=1926, death_year=2022,
            era=Era.CONTEMPORARY,
            traditions=["tradition-zen", "tradition-mahayana", "tradition-buddhism"],
            domains=["domain-theology", "domain-ethics", "domain-philosophy"],
            bio="Vietnamese Zen master of engaged Buddhism and interbeing whose theology IS a torus theology: the sacred is the ordinary surface you already stand on.",
            tier=None,
            manuscript_refs=[
                _R(chapter=18, part="part3-metaphysics", notes="Engaged Buddhism; torus theology"),
            ],
        ),

        # =================================================================
        # ANCIENT / CLASSICAL FIGURES
        # =================================================================

        Thinker(
            id="thinker-plato",
            name="Plato",
            birth_year=-428, death_year=-348,
            era=Era.ANCIENT,
            traditions=["tradition-platonism", "tradition-greek-philosophy"],
            domains=["domain-philosophy", "domain-metaphysics", "domain-epistemology", "domain-political-philosophy"],
            bio="Founder of Western philosophy whose Republic provides the structural blueprint for the AI agent architecture: philosopher-kings, merchants, warriors.",
            tier=None,
            manuscript_refs=[
                _R(chapter=25, part="part4-praxis", notes="Republic mapped onto AI agent architecture"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-socrates", relation_type=_T.STUDENT_OF, description="Plato was Socrates's most famous student"),
                _REL(target_id="thinker-aristotle", relation_type=_T.INFLUENCED, description="Aristotle was Plato's student at the Academy"),
                _REL(target_id="thinker-plotinus", relation_type=_T.INFLUENCED, description="Plotinus revived and developed Platonic philosophy"),
            ],
        ),
        Thinker(
            id="thinker-aristotle",
            name="Aristotle",
            birth_year=-384, death_year=-322,
            era=Era.ANCIENT,
            traditions=["tradition-aristotelianism", "tradition-greek-philosophy"],
            domains=["domain-philosophy", "domain-logic", "domain-metaphysics", "domain-ethics", "domain-science"],
            bio="Systematizer of logic, four causes, and empirical investigation; his physics, economics, and rhetoric provide the applied philosophy framework of Part 6.",
            tier=None,
            manuscript_refs=[
                _R(chapter=39, part="part6-applied-philosophy", notes="Causal physics"),
                _R(chapter=40, part="part6-applied-philosophy", notes="Economics"),
                _R(chapter=41, part="part6-applied-philosophy", notes="Rhetoric"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-plato", relation_type=_T.STUDENT_OF, description="Studied at Plato's Academy for 20 years"),
                _REL(target_id="thinker-aquinas", relation_type=_T.INFLUENCED, description="Aquinas Christianized Aristotelian philosophy"),
            ],
        ),
        Thinker(
            id="thinker-plotinus",
            name="Plotinus",
            birth_year=204, death_year=270,
            era=Era.ANCIENT,
            traditions=["tradition-neoplatonism", "tradition-greek-philosophy"],
            domains=["domain-philosophy", "domain-metaphysics", "domain-mysticism"],
            bio="Founder of Neoplatonism whose doctrine of the One and emanation influenced Christian, Islamic, and Jewish mystical theology.",
            tier=None,
            related_thinkers=[
                _REL(target_id="thinker-plato", relation_type=_T.INFLUENCED, description="Revived and developed Platonic philosophy"),
                _REL(target_id="thinker-augustine", relation_type=_T.INFLUENCED, description="Augustine's theology deeply shaped by Neoplatonism"),
            ],
        ),
        Thinker(
            id="thinker-socrates",
            name="Socrates",
            birth_year=-470, death_year=-399,
            era=Era.ANCIENT,
            traditions=["tradition-greek-philosophy"],
            domains=["domain-philosophy", "domain-ethics", "domain-epistemology"],
            bio="Founder of the examined life and the Socratic method of dialectical inquiry; model for the Digital Socrates AI agent.",
            tier=None,
            manuscript_refs=[
                _R(chapter=25, part="part4-praxis", notes="Model for Digital Socrates agent"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-plato", relation_type=_T.INFLUENCED, description="Plato was his most famous student"),
            ],
        ),

        # =================================================================
        # RELIGIOUS FOUNDERS
        # =================================================================

        Thinker(
            id="thinker-jesus",
            name="Jesus of Nazareth",
            birth_year=-4, death_year=30,
            era=Era.ANCIENT,
            traditions=["tradition-christianity", "tradition-judaism"],
            domains=["domain-theology", "domain-ethics"],
            bio="Historical apocalyptic Jewish prophet whose life, death, and resurrection constitute the central Christ-event; the Logos emerges through apostolic community experience.",
            tier=None,
            manuscript_refs=[
                _R(chapter=15, part="part3-metaphysics", notes="Apostolic realization thesis"),
                _R(chapter=19, part="part3-metaphysics", notes="Cyclical Christ"),
                _R(chapter=21, part="part3-metaphysics", notes="Christ as anti-scapegoat"),
            ],
        ),
        Thinker(
            id="thinker-muhammad",
            name="Muhammad",
            birth_year=570, death_year=632,
            era=Era.MEDIEVAL,
            traditions=["tradition-islam"],
            domains=["domain-theology"],
            bio="Prophet of Islam whose revelation emerged in a milieu saturated with Syriac Christianity; the Quran may critique heresy rather than orthodoxy.",
            tier=None,
            manuscript_refs=[
                _R(chapter=17, part="part3-metaphysics", notes="Quranic context and Syriac Christian milieu"),
            ],
        ),
        Thinker(
            id="thinker-buddha",
            name="Siddhartha Gautama (Buddha)",
            birth_year=-563, death_year=-483,
            era=Era.ANCIENT,
            traditions=["tradition-buddhism"],
            domains=["domain-philosophy", "domain-psychology", "domain-ethics"],
            bio="Founder of Buddhism whose insights into suffering, impermanence, and interdependence provide the cyclical complement to Abrahamic linear eschatology.",
            tier=None,
            manuscript_refs=[
                _R(chapter=18, part="part3-metaphysics", notes="Buddhism expanded"),
                _R(chapter=19, part="part3-metaphysics", notes="Samsara-nirvana synthesis"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-nagarjuna", relation_type=_T.INFLUENCED, description="Nagarjuna systematized Buddhist philosophy centuries later"),
            ],
        ),

        # =================================================================
        # MODERN THINKERS
        # =================================================================

        Thinker(
            id="thinker-teilhard",
            name="Pierre Teilhard de Chardin",
            birth_year=1881, death_year=1955,
            era=Era.MODERN,
            traditions=["tradition-catholic-christianity", "tradition-process-philosophy"],
            domains=["domain-theology", "domain-science"],
            bio="Jesuit paleontologist who developed the Omega Point: evolutionary convergence toward a divine attractor, anticipating the Riemann sphere theology.",
            tier=None,
        ),
        Thinker(
            id="thinker-whitehead",
            name="Alfred North Whitehead",
            birth_year=1861, death_year=1947,
            era=Era.MODERN,
            traditions=["tradition-process-philosophy"],
            domains=["domain-philosophy", "domain-metaphysics", "domain-mathematics"],
            bio="Founder of process philosophy emphasizing becoming over being; reality as process rather than substance aligns with the theology's emergentist ontology.",
            tier=None,
        ),
        Thinker(
            id="thinker-mcgilchrist",
            name="Iain McGilchrist",
            birth_year=1953, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-neuroscience", "domain-philosophy-of-mind", "domain-philosophy"],
            bio="Neuroscientist whose left/right hemisphere thesis in The Master and His Emissary shows how lateralized cognition shapes civilization's relationship to reality.",
            tier=None,
            manuscript_refs=[
                _R(chapter=1, part="part1-psychology", notes="Left/right hemisphere and consciousness"),
            ],
        ),
        Thinker(
            id="thinker-jaynes",
            name="Julian Jaynes",
            birth_year=1920, death_year=1997,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-psychology", "domain-philosophy-of-mind", "domain-history"],
            bio="Psychologist who proposed the bicameral mind hypothesis: human self-awareness emerged roughly 3000-1000 BCE, aligning with Hebrew biblical chronology.",
            tier=None,
            manuscript_refs=[
                _R(chapter=13, part="part3-metaphysics", notes="Breakdown of bicameral mind as consciousness emergence"),
            ],
        ),
        Thinker(
            id="thinker-vervaeke",
            name="John Vervaeke",
            birth_year=1966, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-cognitive-psychology", "domain-philosophy", "domain-comparative-religion"],
            bio="Cognitive scientist whose Awakening from the Meaning Crisis lectures identify the meta-crisis underlying all societal crises: civilization without orientation.",
            tier=None,
            manuscript_refs=[
                _R(chapter=38, part="part5-apostolic-agenda", notes="The meaning crisis as meta-crisis"),
            ],
        ),
        Thinker(
            id="thinker-kotler",
            name="Steven Kotler",
            birth_year=1967, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-neuroscience", "domain-psychology"],
            bio="Flow researcher whose work with Parvizi-Wayne and Friston grounds the theology's prophetic states in active inference neuroscience.",
            tier=None,
            manuscript_refs=[
                _R(chapter=9, part="part2-epistemology", notes="Flow and intuition papers"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-friston", relation_type=_T.COLLABORATED, description="Co-authored flow and intuition papers"),
                _REL(target_id="thinker-csikszentmihalyi", relation_type=_T.INFLUENCED, description="Built on Csikszentmihalyi's original flow research"),
            ],
        ),
        Thinker(
            id="thinker-csikszentmihalyi",
            name="Mihaly Csikszentmihalyi",
            birth_year=1934, death_year=2021,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-psychology", "domain-cognitive-psychology"],
            bio="Originator of flow state research, identifying the optimal experience zone between anxiety and boredom as key to human flourishing.",
            tier=None,
            related_thinkers=[
                _REL(target_id="thinker-kotler", relation_type=_T.INFLUENCED, description="Kotler extended Csikszentmihalyi's flow research"),
            ],
        ),
        Thinker(
            id="thinker-metzinger",
            name="Thomas Metzinger",
            birth_year=1958, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-philosophy-of-mind", "domain-neuroscience"],
            bio="Philosopher of mind whose self-model theory and minimal phenomenal selfhood explain how narrative self dissolves in flow and mystical states.",
            tier=None,
        ),
        Thinker(
            id="thinker-hohwy",
            name="Jakob Hohwy",
            birth_year=None, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-philosophy-of-mind", "domain-neuroscience"],
            bio="Philosopher and neuroscientist of predictive processing whose The Predictive Mind provides the cognitive architecture underlying active inference accounts of consciousness.",
            tier=None,
            related_thinkers=[
                _REL(target_id="thinker-friston", relation_type=_T.INFLUENCED, description="Predictive processing built on Friston's free energy principle"),
            ],
        ),

        # =================================================================
        # ECONOMICS & POLITICAL THEORY
        # =================================================================

        Thinker(
            id="thinker-hayek",
            name="Friedrich Hayek",
            birth_year=1899, death_year=1992,
            era=Era.CONTEMPORARY,
            traditions=["tradition-vienna-circle"],
            domains=["domain-economics", "domain-political-philosophy"],
            bio="Austrian economist whose spontaneous order concept is emergence applied to markets, supporting the theology's decentralized ecclesiology.",
            tier=None,
            manuscript_refs=[
                _R(chapter=12, part="part2-epistemology", notes="Austrian economics: spontaneous order as emergence"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-popper", relation_type=_T.CONTEMPORARY_OF, description="Both Vienna Circle alumni; mutual influence on open society thinking"),
            ],
        ),
        Thinker(
            id="thinker-acemoglu",
            name="Daron Acemoglu",
            birth_year=1967, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-economics", "domain-political-philosophy"],
            bio="Institutional economist whose inclusive vs extractive institutions framework maps onto the normie/psycho/schizo taxonomy at institutional scale.",
            tier=None,
            manuscript_refs=[
                _R(chapter=12, part="part2-epistemology", notes="Institutional economics: inclusive vs extractive"),
            ],
        ),
        Thinker(
            id="thinker-cowen",
            name="Tyler Cowen",
            birth_year=1962, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=["tradition-progress-studies"],
            domains=["domain-economics"],
            bio="Economist whose Great Stagnation thesis frames current stagnation as Kuhnian late-paradigm crisis requiring new frameworks.",
            tier=None,
            manuscript_refs=[
                _R(chapter=12, part="part2-epistemology", notes="Progress Studies: stagnation as paradigm crisis"),
            ],
        ),
        Thinker(
            id="thinker-piketty",
            name="Thomas Piketty",
            birth_year=1971, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-economics", "domain-history"],
            bio="Economist whose r > g formula is a correlational observation; the causal mechanism requires adding regulatory capture, information asymmetry, and structural power.",
            tier=None,
            manuscript_refs=[
                _R(chapter=34, part="part5-apostolic-agenda", notes="Piketty's r > g as correlational; causal mechanism deeper"),
            ],
        ),
        Thinker(
            id="thinker-plekhanov",
            name="Georgi Plekhanov",
            birth_year=1856, death_year=1918,
            era=Era.MODERN,
            traditions=[],
            domains=["domain-political-philosophy", "domain-history"],
            bio="Marxist theorist of structural contingency: structure is determined but instantiation is contingent, providing the free will synthesis at historical scale.",
            tier=None,
            manuscript_refs=[
                _R(chapter=26, part="part4-praxis", notes="Core of free will / determinism synthesis"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-marx", relation_type=_T.STUDENT_OF, description="Plekhanov developed and applied Marx's historical materialism"),
            ],
        ),
        Thinker(
            id="thinker-marx",
            name="Karl Marx",
            birth_year=1818, death_year=1883,
            era=Era.MODERN,
            traditions=[],
            domains=["domain-political-philosophy", "domain-economics", "domain-sociology"],
            bio="Philosopher whose critique of capitalism was largely correct but whose assumption that correct social order would emerge naturally from destruction was fatally flawed.",
            tier=None,
            manuscript_refs=[
                _R(chapter=28, part="part4-praxis", notes="Kirill Principle: Marxism's fatal flaw was construction-before-destruction"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-hegel", relation_type=_T.INFLUENCED, description="Inverted Hegel's dialectic from idealist to materialist"),
                _REL(target_id="thinker-plekhanov", relation_type=_T.INFLUENCED, description="Plekhanov applied Marxist historical materialism"),
            ],
        ),

        # =================================================================
        # LITERATURE & ARTS
        # =================================================================

        Thinker(
            id="thinker-milton",
            name="John Milton",
            birth_year=1608, death_year=1674,
            era=Era.EARLY_MODERN,
            traditions=["tradition-protestant-christianity"],
            domains=["domain-arts", "domain-theology"],
            bio="Poet of Paradise Lost whose Satan serves as the engine of the dialectic: does not derail God's plan but initiates it through the Fall.",
            tier=None,
            manuscript_refs=[
                _R(chapter=14, part="part3-metaphysics", notes="Milton's Satan as dialectic engine; Fall as felix culpa"),
            ],
        ),
        Thinker(
            id="thinker-herbert",
            name="Frank Herbert",
            birth_year=1920, death_year=1986,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-arts", "domain-political-philosophy"],
            bio="Author of Dune whose warning about charismatic leaders and unforkable power structures provides the structural safeguard against the theology's own failure mode.",
            tier=None,
            manuscript_refs=[
                _R(chapter=27, part="part4-praxis", notes="The Dune warning: understanding predator logic without adopting it"),
            ],
        ),
        Thinker(
            id="thinker-gibran",
            name="Kahlil Gibran",
            birth_year=1883, death_year=1931,
            era=Era.MODERN,
            traditions=["tradition-maronite"],
            domains=["domain-arts", "domain-philosophy"],
            bio="Lebanese-American poet and artist whose Jesus the Son of Man offers a literary, multi-perspectival account of Christ from the Eastern Christian tradition.",
            tier=None,
        ),
        Thinker(
            id="thinker-wilde",
            name="Oscar Wilde",
            birth_year=1854, death_year=1900,
            era=Era.MODERN,
            traditions=[],
            domains=["domain-arts", "domain-aesthetics"],
            bio="Writer and aestheticist who showed appearance IS reality's surface dimension; attending to appearance is a form of intelligence, not vanity.",
            tier=None,
            manuscript_refs=[
                _R(chapter=23, part="part3-metaphysics", notes="Aesthetics: care/composition as cognitive operation"),
            ],
        ),
        Thinker(
            id="thinker-schopenhauer",
            name="Arthur Schopenhauer",
            birth_year=1788, death_year=1860,
            era=Era.MODERN,
            traditions=["tradition-german-idealism"],
            domains=["domain-philosophy", "domain-metaphysics", "domain-aesthetics"],
            bio="Philosopher of Will whose aesthetic theory shows art as epistemic access to Platonic Ideas, bypassing the Will's endless striving, mapping onto Pirsig's Quality.",
            tier=None,
            manuscript_refs=[
                _R(chapter=23, part="part3-metaphysics", notes="Aesthetic experience as epistemology"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-nietzsche", relation_type=_T.INFLUENCED, description="Nietzsche's early thought deeply shaped by Schopenhauer"),
            ],
        ),

        # =================================================================
        # ADDITIONAL IMPORTANT THINKERS
        # =================================================================

        Thinker(
            id="thinker-yudkowsky",
            name="Eliezer Yudkowsky",
            birth_year=1979, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=["tradition-rationalist-community"],
            domains=["domain-ai-ml", "domain-philosophy", "domain-epistemology"],
            bio="AI safety researcher and founder of LessWrong whose rationalist community represents the theology's closest intellectual cousins, lacking only a telos.",
            tier=None,
            manuscript_refs=[
                _R(chapter=12, part="part2-epistemology", notes="Rationality community: closest cousins"),
            ],
        ),
        Thinker(
            id="thinker-scott-alexander",
            name="Scott Alexander",
            birth_year=None, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=["tradition-rationalist-community"],
            domains=["domain-philosophy", "domain-social-science"],
            bio="Rationalist writer whose Meditations on Moloch identifies the coordination failure that IS the antichrist structure described in the theology.",
            tier=None,
            manuscript_refs=[
                _R(chapter=12, part="part2-epistemology", notes="MOLOCH = the antichrist structure"),
            ],
        ),
        Thinker(
            id="thinker-macaskill",
            name="William MacAskill",
            birth_year=1987, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=["tradition-ea"],
            domains=["domain-ethics", "domain-philosophy"],
            bio="Co-founder of Effective Altruism movement whose longtermism converges with eschatology but rests on motivationally thin utilitarian foundations.",
            tier=None,
            manuscript_refs=[
                _R(chapter=12, part="part2-epistemology", notes="EA critique: utilitarian foundation, Pearl Level 1-2 only"),
            ],
        ),
        Thinker(
            id="thinker-sunstein",
            name="Cass Sunstein",
            birth_year=1954, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=["tradition-behavioral-economics"],
            domains=["domain-economics", "domain-political-philosophy"],
            bio="Legal scholar and behavioral economist who co-authored Nudge; the theology's ethical criterion for nudging is transparency.",
            tier=None,
            manuscript_refs=[
                _R(chapter=12, part="part2-epistemology", notes="Nudge theory: ethical criterion is transparency"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-thaler", relation_type=_T.COLLABORATED, description="Co-authored Nudge"),
            ],
        ),
        Thinker(
            id="thinker-thaler",
            name="Richard Thaler",
            birth_year=1945, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=["tradition-behavioral-economics"],
            domains=["domain-economics", "domain-cognitive-psychology"],
            bio="Behavioral economist who co-authored Nudge and demonstrated systematic irrationality in economic decision-making.",
            tier=None,
            manuscript_refs=[
                _R(chapter=12, part="part2-epistemology", notes="Nudge theory"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-kahneman", relation_type=_T.INFLUENCED, description="Built behavioral economics on Kahneman's cognitive bias research"),
                _REL(target_id="thinker-sunstein", relation_type=_T.COLLABORATED, description="Co-authored Nudge"),
            ],
        ),
        Thinker(
            id="thinker-frankopan",
            name="Peter Frankopan",
            birth_year=1971, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-history"],
            bio="Historian of the Silk Roads who reveals how convergences in world wisdom traditions were produced by actual historical networks of exchange.",
            tier=None,
            manuscript_refs=[
                _R(chapter=18, part="part3-metaphysics", notes="Silk Roads as original Republic of Letters"),
            ],
        ),
        Thinker(
            id="thinker-griffith",
            name="Sidney Griffith",
            birth_year=None, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=["tradition-syriac-christianity"],
            domains=["domain-theology", "domain-history", "domain-comparative-religion"],
            bio="Scholar of The Church in the Shadow of the Mosque who demonstrates the Syriac Christian context essential to understanding the Quran.",
            tier=None,
            manuscript_refs=[
                _R(chapter=17, part="part3-metaphysics", notes="Syriac Christian context of the Quran"),
            ],
        ),
        Thinker(
            id="thinker-reynolds",
            name="Gabriel Said Reynolds",
            birth_year=None, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-theology", "domain-biblical-studies", "domain-comparative-religion"],
            bio="Quranic scholar who analyzes the Quran's biblical subtext, showing deep intertextual connections between Quranic and biblical narratives.",
            tier=None,
            manuscript_refs=[
                _R(chapter=17, part="part3-metaphysics", notes="Quran and its biblical subtext"),
            ],
        ),
        Thinker(
            id="thinker-shah-kazemi",
            name="Reza Shah-Kazemi",
            birth_year=None, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=["tradition-sufism", "tradition-islam"],
            domains=["domain-comparative-religion", "domain-theology"],
            bio="Comparative theologian whose The Other in the Light of the One shows convergence between Islamic, Christian, and Hindu mystical traditions.",
            tier=None,
            manuscript_refs=[
                _R(chapter=17, part="part3-metaphysics", notes="Islamic-Christian comparative theology"),
            ],
        ),
        Thinker(
            id="thinker-scarry",
            name="Elaine Scarry",
            birth_year=1946, death_year=None,
            era=Era.CONTEMPORARY,
            traditions=[],
            domains=["domain-literary-theory", "domain-aesthetics", "domain-ethics"],
            bio="Literary theorist whose On Beauty and Being Just argues that beauty leads to justice, connecting aesthetics to ethics.",
            tier=None,
            manuscript_refs=[
                _R(chapter=23, part="part3-metaphysics", notes="Beauty and justice connection"),
            ],
        ),
        Thinker(
            id="thinker-fichte",
            name="Johann Gottlieb Fichte",
            birth_year=1762, death_year=1814,
            era=Era.EARLY_MODERN,
            traditions=["tradition-german-idealism"],
            domains=["domain-philosophy", "domain-metaphysics", "domain-epistemology"],
            bio="German Idealist who formalized self-composition as the fundamental act of freedom: identity is performed through every choice, including aesthetic ones.",
            tier=None,
            manuscript_refs=[
                _R(chapter=23, part="part3-metaphysics", notes="Self-composition as freedom"),
            ],
            related_thinkers=[
                _REL(target_id="thinker-hegel", relation_type=_T.INFLUENCED, description="Hegel developed Fichte's thesis-antithesis-synthesis"),
            ],
        ),

        # =================================================================
        # SUPPLEMENTARY — Not listed above but referenced in CLAUDE.md
        # =================================================================

        Thinker(
            id="thinker-kierkegaard",
            name="Soren Kierkegaard",
            birth_year=1813, death_year=1855,
            era=Era.MODERN,
            traditions=["tradition-existentialism", "tradition-christianity"],
            domains=["domain-philosophy", "domain-theology", "domain-ethics"],
            bio="Father of existentialism who insisted on the irreducibility of individual existence against Hegel's totalizing system.",
            tier=None,
            related_thinkers=[
                _REL(target_id="thinker-hegel", relation_type=_T.OPPOSED, description="Reacted against Hegel's rationalist system"),
                _REL(target_id="thinker-sartre", relation_type=_T.INFLUENCED, description="Foundational influence on existentialism"),
            ],
        ),
    ]
