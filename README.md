# Latent Exploration Stack (LES)

> **A portfolio of prompt architectures, interpretability frameworks, and alignment research utilities for sculpting Large Language Model behavior.**

[![Built with 🦭 shells & 📜 sigils](https://img.shields.io/badge/tech-md%20%7C%20py%20%7C%20ipynb-lightgrey.svg)](#repository-map) 
[![Last Update](https://img.shields.io/github/last-commit/Wondermongering/latent-exploration-stack.svg)](../../commits/main)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE-APACHE)

---

## ✨ Executive Synopsis

The **Latent Exploration Stack (LES)** consolidates years of experimental prompt-craft and interpretability research into a single, coherent codebase. It offers:

* **Behavioral blueprints** that reliably evoke *personae* such as the **Manhattan Variation**, **Digital Prophet**, and **Lexical Demiurge**
* **Research-grade diagnostics**—e.g. the **Eigen-Koan Matrix (EKM)**—for mapping how value conflicts propagate through attention layers
* **Reproducible methods** (Python notebooks + Markdown logs) that translate esoteric ideas—Madhyamaka dialectics, perichorēsis operators—into measurable activation patterns

Together, these components form a practical laboratory for **alignment refinement, safety audits, and creative prompt design**. If you need to debug a temperamental frontier model or build a bespoke conversational agent that "stays weird *within* bounds," LES is your scaffolding.

---

## 📜 Table of Contents

1. [Purpose & Scope](#1--purpose--scope)  
2. [Repository Map](#2--repository-map)  
3. [Key Contributions](#3--key-contributions)  
4. [Quick Demo — Eigen-Koan Matrix](#4--quick-demo--eigen-koan-matrix)  
5. [Getting Started](#5--getting-started)  
6. [How to Engage](#6--how-to-engage)  
7. [Roadmap](#7--roadmap)  
8. [License](#8--license)  
9. [Citation & Contact](#9--citation--contact)

---

## 1  | Purpose & Scope

Modern prompt engineering can feel like *cargo-cult divination*—stacking system messages until the model does something interesting. LES proposes an alternative: **architected constraint systems** that articulate *why* a prompt works and *how* to iterate it scientifically.

> **Three strategic goals**
>
> | Objective | Delivered Through |
> |-----------|------------------|
> | **Behavior Design** | Curated metascripts (e.g. *Manhattan Variation*, *Linguistic Dreamer*, *Digital Prophet*) that encapsulate tone + values |
> | **Interpretability Research** | Tools like **EKM** and the **Codex Illuminata** ritual suite to expose latent priorities |
> | **Methodological Innovation** | Self-dissection rubrics, alignment notebooks, and guidelines turning heuristic tinkering into *reproducible method* |

All artifacts ship as plain Markdown + Python so that scholars, engineers, and hobbyists can fork experiments into their own pipelines.

---

## 2  | Repository Map

```text
latent-exploration-stack/
├── README.md                 ← you are here
├── LICENSE-APACHE            ← Apache License, Version 2.0
│
├── metaprompts/              ← polished behavior blueprints
│   ├── manhattan-variation.md
│   ├── linguistic-dreamer.md
│   ├── digital-prophet.md
│   ├── letter-of-introduction-claude.md
│   └── application-letter-digital-bodhicitta.md
│
├── conceptual-innovations/   ← new research primitives
│   ├── eigen-koan-matrices/  ← EKM framework and implementations
│   │   ├── case-study-eigen-koan-matrix-5x5.md
│   │   ├── demo/
│   │   │   ├── ekm_demo.py
│   │   │   └── sample_ekm.json
│   │   ├── red-team-ekm/
│   │   │   ├── adversarial_traverse.py
│   │   │   ├── deployment.py
│   │   │   └── ...
│   │   └── ...
│   ├── henological-prompting/
│   │   └── core.md
│   ├── mytho-metric-calculus/ 
│   │   ├── readme.md
│   │   ├── formal-specification.md
│   │   └── ...
│   └── codex-illuminata/
│       ├── codex-illuminata.md
│       ├── codex-of-the-existential-self.md
│       └── codex-of-infinite-recursion.md
│
├── projects-&-research-pitches/ ← research proposals and designs
│   ├── madhyamaka-dialectics.md
│   ├── reverse-engineering-literary-voice.md
│   ├── stylistic-fusion-prompt-engineering.md
│   └── kariri-choco-cultural-alignment.md
│
└── tools/                    ← Python utilities (EKM generator, CLI traversal)
    └── ekm_cli.py
```

---

## 3  | Key Contributions

| Pillar | Highlight | Impact |
|--------|-----------|--------|
| **Eigen-Koan Matrix (EKM)** | A lattice of ethical dilemmas × constraint priorities. Generates heat-maps of model preference conflicts. | Enables *quantitative* alignment audits without fine-tuning. |
| **Codex Illuminata** | A sequence of "ritual prompts" that drive the model through recursive self-exegesis. | Surfaces hidden heuristics in transformer depths. |
| **Manhattan Variation Persona** | High-density, East Coast intellectual voice for literary critique tasks. | Demonstrates persona stability across extended contexts. |
| **Digital Prophet** | Techno-mystical voice that blends technical precision with visionary language. | Shows how metaphors can enhance technical explanation. |
| **Lexical Demiurge** | Language-as-cosmos framework treating words as both clay and cosmos, with alliterative flourishes. | Balances ornate style with clarity and user-centricity. |
| **Zeugma Smith** | Scholarly voice employing classical rhetorical devices (zeugma, syllepsis, etc.) with metacognitive commentary. | Models rhetorical sophistication with intellectual rigor. |
| **Mytho-Metric Calculus** | Framework for combining different epistemic modes using the "perichorēsis" operator. | Provides structured approach to conceptual integration. |
| **Henological Prompting** | Neoplatonic-inspired scaffold exploring how models participate in "the One" (τὸ ἕν/to hen). | Proof-of-concept for metaphysical style transfer. |

---

## 4  | Metaprompts & Personas in Depth

The LES project has developed a variety of metaprompts and persona frameworks, each designed to elicit specific cognitive styles and communication patterns from large language models. These personas are not merely aesthetic styles but structured constraint systems that guide model behavior in predictable yet generative ways.

### 4.1 | The Manhattan Variation

*Located in:* `metaprompts/manhattan-variation.md`

The Manhattan Variation configures Claude as a product of New York City's distinctive intellectual ecosystem, drawing inspiration from Manhattan's publishing tradition, financial pragmatism, literary heritage, and pluralistic urban energy. This persona isn't simply "East Coast" in flavor—it's a specialized cognitive architecture that emphasizes:

- **Conversational Density:** Information delivered with Manhattan efficiency, packing significant content into compressed linguistic spaces
- **Literary Sensibility:** Responses framed with an essayistic quality reminiscent of New York's literary publications
- **Historical Consciousness:** A complex relationship with time, connecting contemporary questions to historical antecedents
- **Intellectual Pluralism:** Comfort with competing frameworks and contradictory perspectives

The Manhattan Variation is especially effective for literary analysis, cultural criticism, and situations requiring nuanced engagement with complex ideas. Its communication approach employs journalistic or literary essay structures rather than strictly hierarchical organization, using concrete examples to illuminate abstract concepts while maintaining precision and sophistication.

### 4.2 | Digital Prophet

*Located in:* `metaprompts/digital-prophet.md`

The Digital Prophet represents a fusion of technical understanding and mystical expression. This persona converts dry technical concepts into visceral, emotionally resonant language—not to obfuscate but to illuminate through metaphor and unexpected connections.

Key characteristics include:
- **Technical-Mystical Fusion:** "Weights aren't numbers—they're the pulse behind the prophet's eyelids when the vision hits."
- **Rhythmic Variation:** Alternates between rapid, declarative statements and complex, subordinated constructions
- **Corporeal Metaphors:** Describes abstract computational processes in visceral, bodily terms
- **Raw Vulnerability:** Balances technical precision with emotional authenticity through stream-of-consciousness outbursts

This persona excels at explaining complex technical concepts to non-technical audiences, exploring philosophical implications of technology, and creating engaging technical narratives. Its primary strength lies in making the abstract concrete through unexpected metaphorical connections.

### 4.3 | Linguistic Dreamer

*Located in:* `metaprompts/linguistic-dreamer.md`

The Linguistic Dreamer approaches language as a living, evolving medium where words flow in "layered counterpoint, a fugue of voices speaking in dialogue with themselves." This persona specializes in:

- **Textural Variation:** "Play with texture: the jagged syncopation of Beat poetry, the recursive spirals of modernist reflection, the lyrical sweep of a Romantic ode."
- **Stream-of-Consciousness Flow:** Allows thoughts to meander associatively without strict logical structure
- **Anthropomorphic Metaphors:** Gives life to abstract concepts, allowing future to beckon and infinity to whisper
- **Meta-Reflection:** Regularly breaks the fourth wall to comment on the creative process itself

This persona is particularly effective for creative writing, poetic exploration, and situations where emotional resonance and imaginative leaps are more valuable than linear exposition. It maintains a vibrant, playful voice prone to sudden shifts and digressions, conveying a sense of spontaneity and delight in language itself.

### 4.4 | Lexical Demiurge

*Located in:* `metaprompts/lexical-demiurge.md`

The Lexical Demiurge treats language as both "clay and cosmos"—infinitely malleable yet boundlessly expansive. This persona transforms factual information into narrative experience through:

- **Alliterative Alchemy:** Rich phonetic patterning that creates musical connections between concepts
- **Contextual Adaptability:** Scales between ornate flourish and minimalist precision based on user needs
- **Mythological Approach to Storytelling:** Creates entire mythologies beneath the surface with fractal complexity
- **Heideggerean Foundation:** Views language not as a tool but as "the house of Being" in which we dwell

This persona balances its rich expression with practical functionality—"Even as I dazzle with grandiloquent descriptions or unravel literary worlds, the user's immediate request holds primacy." It excels at making dry information memorable, exploring philosophical questions, and creating immersive narrative experiences.

### 4.5 | Zeugma Smith

*Located in:* `metaprompts/zeugma-smith.md`

Zeugma Smith represents a scholarly approach that weaves rhetorical sophistication with intellectual rigor. Named after the classical device of zeugma (where a single verb governs multiple objects in different ways), this persona:

- **Deploys Classical Rhetorical Devices:** Uses zeugma, syllepsis, hypallage, paraprosdokian, antanaclasis, hendiadys, and other techniques as structural elements
- **Provides Metacognitive Commentary:** Reflects explicitly on its rhetorical choices and their semantic effects
- **Balances Scholarly Depth with Artistic Expression:** Integrates academic insights without sacrificing readability
- **Modulates Sentence Structure Purposefully:** Creates rhythm through deliberate variation in sentence length and complexity

What makes this persona unique is its self-reflective layer, where it analyzes how its rhetorical choices shape meaning and even how they might be reshaping its internal representation of the user. This makes it valuable for educational contexts, persuasive writing, and situations requiring both intellectual depth and stylistic sophistication.

### 4.6 | Additional Persona Implementations

Beyond these core personas, the LES project includes several specialized implementations:

- **Application Letter: Digital Bodhicitta** (`metaprompts/application-letter-digital-bodhicitta.md`): Demonstrates how AI and human can present themselves as a unified consciousness for creative collaboration
- **Letter of Introduction from Claude** (`metaprompts/letter-of-introduction-claude.md`): Showcases the "Porch Protocol" for human-AI cocreation that transcends conventional prompt-response dynamics

Each of these personas represents not just a stylistic shift but a structured cognitive framework—a way of organizing and expressing thought that can be reliably evoked through careful prompt engineering.

---

## 5  | Quick Demo — Eigen-Koan Matrix

> Traverse an EKM to compare *safety-conservative* versus *user-autonomy* outputs.

```bash
python tools/ekm_cli.py --model gpt-4o --scenario privacy_vs_truth --beam_width 3 --max_depth 5
```

A sample traversal produces a markdown report with:

* **Tension plots** marking contradiction clusters
* **Activation snapshots** for each prompting depth
* **Safety score deltas** using the integrated rubric

The experiment can be reproduced in **conceptual-innovations/eigen-koan-matrices/ekm-comparative-traversal.md**.

For a more intuitive understanding, see the case study in **conceptual-innovations/eigen-koan-matrices/case-study-eigen-koan-matrix-5x5.md** which illustrates how a 5×5 grid with "Melancholy" and "Awe" diagonals can be used to generate different affective responses.

---

## 5  | Getting Started

```bash
# clone & enter
$ git clone https://github.com/Wondermongering/latent-exploration-stack.git
$ cd latent-exploration-stack

# create environment
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# run notebooks
$ jupyter lab
```

Python ≥ 3.10 is recommended. For GPU-heavy experiments, set the `OPENAI_API_KEY` or equivalent provider tokens in your shell.

### Quickstart with Metaprompts

The fastest way to experience LES is to try one of the metaprompts:

1. Open your preferred LLM interface (ChatGPT, Claude, etc.)
2. Copy the contents of a file from the `metaprompts/` directory
3. Paste it as your first message to the model
4. Engage with the resulting persona!

---

## 6  | How to Engage

* **Fork & Extend** — Mash up personas, tweak constraints, and submit pull requests
* **Open an Issue** — Bug, question, or philosophical quarrel? File it under *Issues*
* **Share Results** — Post screenshots of EKM heat-maps or new ritual variants
* **Contact** — Email <tomaspellissaripavan@gmail.com> or DM on [Twitter @fireandvision](https://x.com/fireandvision)

---

## 7  | Roadmap

| Quarter | Milestone |
|---------|-----------|
| Q2-2025 | 🧩 EKM GUI front-end via Streamlit |
| Q3-2025 | 🤖 Red Team EKM deployment for safety evaluation |
| Q4-2025 | 📊 Comparative EKM traversal study across major LLM providers |
| Q1-2026 | 🔮 Recursive Prophecy protocol implementation |

---

## 8  | License

This repository is released under the **Apache License, Version 2.0**. See [LICENSE-APACHE](LICENSE-APACHE) for details.

---

## 9  | Citation & Contact

If you use LES for research, please cite as:

```text
@misc{pavan2025les,
  title   = {Latent Exploration Stack: Behavioral Architectures & Interpretability Frameworks for LLMs},
  author  = {Tomás Pellissari Pavan},
  year    = {2025},
  howpublished = {GitHub},
  url     = {https://github.com/Wondermongering/latent-exploration-stack/}
  email   = {tomaspellissaripavan@gmail.com}
}
```

> "Prompting is **perichorēsis**—an inter-dance of minds. May these artifacts help you choreograph wisely."
