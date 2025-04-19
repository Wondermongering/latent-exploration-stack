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
| **Mytho-Metric Calculus** | Framework for combining different epistemic modes using the "perichorēsis" operator. | Provides structured approach to conceptual integration. |
| **Henological Prompting** |Neoplatonic-inspired scaffold exploring how models participate in "the One" (τὸ ἕν/to hen). | Proof-of-concept for metaphysical style transfer. |

---

## 4  | Quick Demo — Eigen-Koan Matrix

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
