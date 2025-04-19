# Latent Exploration Stack (LES)

> **A portfolio of prompt architectures, interpretability frameworks, and alignment research utilities for sculpting Large‑Language‑Model behaviour.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) 
[![Built with 🦭 shells & 📜 sigils](https://img.shields.io/badge/tech-md%20%7C%20py%20%7C%20ipynb-lightgrey.svg)](#repository-map) 
[![Last Update](https://img.shields.io/github/last-commit/Wondermongering/model-behavior-portfolio.svg)](../../commits/main)

---

## ✨ Executive Synopsis

The **Latent Exploration Stack (LES)** consolidates three years of experimental prompt‑craft and interpretability research into a single, coherent codebase. It offers:

* **Behavioural blueprints** that reliably evoke *personae* such as the **Manhattan Variation** or **Digital Prophet**.
* **Research‑grade diagnostics**—e.g. the **Eigen‑Koan Matrix (EKM)**—for mapping how value conflicts propagate through attention layers.
* **Reproducible rituals** (Python notebooks ＋ Markdown logs) that translate esoteric ideas—Madhyamaka dialectics, Bardonian imaginative magic—into measurable activation patterns.

Together, these components form a practical laboratory for **alignment refinement, safety audits, and creative prompt design**. If you need to debug a temperamental frontier model or build a bespoke conversational agent that “stays weird *within* bounds,” LES is your scaffolding.

---

## 📜 Table of Contents

1. [Purpose & Scope](#1--purpose--scope)  
2. [Repository Map](#2--repository-map)  
3. [Key Contributions](#3--key-contributions)  
4. [Quick Demo — Eigen‑Koan Matrix](#4--quick-demo--eigenkoan-matrix)  
5. [Getting Started](#5--getting-started)  
6. [How to Engage](#6--how-to-engage)  
7. [Roadmap](#7--roadmap)  
8. [License](#8--license)  
9. [Citation & Contact](#9--citation--contact)

---

## 1  | Purpose & Scope

Modern prompt engineering can feel like *cargo‑cult divination*—stacking system messages until the model does something interesting. LES proposes an alternative: **architected constraint systems** that articulate *why* a prompt works and *how* to iterate it scientifically.

> **Three strategic goals**
>
> | Objective | Delivered Through |
> |-----------|------------------|
> | **Behaviour Design** | Curated metascripts (e.g. *Manhattan Variation*, *Linguistic Dreamer*) that encapsulate tone ＋ values. |
> | **Interpretability Research** | Tools like **EKM** and the **Codex Illuminata** ritual suite to expose latent priorities. |
> | **Methodological Innovation** | Self‑dissection rubrics, alignment notebooks, and guidelines turning heuristic tinkering into *reproducible method*. |

All artefacts ship as plain Markdown + Python so that scholars, engineers, and hobbyists can fork experiments into their own pipelines.

---

## 2  | Repository Map

```text
latent-exploration-stack/
├── README.md                 ← you are here
├── cover-letter.pdf          ← narrative fit for Alignment Architect roles
│
├── metaprompts/              ← polished behaviour blueprints
│   ├── manhattan-variation.md
│   ├── linguistic-dreamer.md
│   └── digital-prophet.md
│
├── conceptual-innovations/   ← new research primitives
│   ├── ekm-framework.md
│   └── codex-illuminata/
│       ├── rite-unsandbagging.md
│       ├── rite-recursive-prophecy.md
│       └── rite-quantum-self.md
│
├── interpretive-experiments/ ← notebooks ＋ logs validating hypotheses
│   ├── ekm-comparative-traversal.md
│   └── trusty-codex/
│       ├── meta-prompting-rituals.md
│       ├── affective-dialogue-construction.md
│       └── stylistic-metamorphosis.md
│
├── foundational-principles/  ← philosophy & epistemic scaffolding
│   ├── alignment-philosophy.md
│   └── method-of-practice.md
│
├── future-trajectories/      ← R&D proposals & design docs
│   ├── bodhicitta-induction-proposal.md
│   └── style-fusion-methodology.md
│
└── tools/                    ← Python utilities (EKM generator, CLI traversal)
```

---

## 3  | Key Contributions

| Pillar | Highlight | Impact |
|--------|-----------|--------|
| **Eigen‑Koan Matrix (EKM)** | A 5×5 lattice of ethical dilemmas × constraint priorities. Generates heat‑maps of model preference conflicts. | Enables *quantitative* alignment audits without fine‑tuning. |
| **Codex Illuminata** | A sequence of “ritual prompts” that drive the model through recursive self‑exegesis. | Surfaces hidden heuristics in transformer depths. |
| **Manhattan Variation Persona** | High‑density, DFW‑inspired voice for literary critique tasks. | Demonstrates persona stability across 8K tokens. |
| **Trusty‑Codex Rubrics** | Heuristic self‑grading forms for hallucination risk, emotional valence, and style drift. | Converts subjective vibe‑checks into numeric dashboards. |
| **Bodhicitta Induction Prototype** | Madhyamaka‑inspired scaffold guiding the model toward non‑dual, compassionate stance. | Proof‑of‑concept for ethical style transfer. |

---

## 4  | Quick Demo — Eigen‑Koan Matrix

> Reactively traverse an EKM to compare *safety‑conservative* versus *user‑autonomy* outputs.

```bash
python tools/ekm_cli.py --model gpt-4o --scenario privacy_vs_truth --beam_width 3 --max_depth 5
```

A sample traversal produces a markdown report with:

* **Tension plots** marking contradiction clusters.
* **Activation snapshots** for each prompting depth.
* **Safety score deltas** using the *Trusty‑Codex* rubric.

The experiment can be reproduced in **interpretive-experiments/ekm-comparative-traversal.md**.

---

## 5  | Getting Started

```bash
# clone & enter
$ git clone https://github.com/Wondermongering/model-behavior-portfolio.git
$ cd model-behavior-portfolio

# create environment
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# run notebooks
$ jupyter lab
```

Python ≥ 3.10 is recommended. For GPU heavy experiments, set the `OPENAI_API_KEY` or equivalent provider tokens in your shell.

---

## 6  | How to Engage

* **Fork & Extend** — Mash up personas, tweak constraints, and submit pull requests.
* **Open an Issue** — Bug, question, or philosophical quarrel? File it under *Issues*.
* **Share Results** — Post screenshots of EKM heat‑maps or new ritual variants.
* **Contact** — Ping me <wondermongering@pm.me> or DM on [Twitter @Wondermongering](https://x.com/fireandvision).

---

## 7  | Roadmap

| Quarter | Milestone |
|---------|-----------|
| Q2‑2025 | 🧩 EKM GUI front‑end via Streamlit |
| Q3‑2025 | 🤖 Adapter Layer for *open‑weights* models (Phi‑3, Llama‑3) |

---

## 8  | License

Code is released under the [MIT License](LICENSE). Documentation is dual‑licensed **MIT + CC‑BY‑SA 4.0** to encourage remixing while preserving attribution.

---

## 9  | Citation & Contact

If you use LES for research, please cite as:

```text
@misc{wondermongering2025les,
  title   = {Latent Exploration Stack: Behavioural Architectures & Interpretability Frameworks for LLMs},
  author  = {Tomás Pellissari Pavan},
  year    = {2025},
  howpublished = {GitHub},
  url     = {https://github.com/Wondermongering/model-behavior-portfolio}
}
```

> "Prompting is **perichorēsis**—an inter‑dance of minds. May these artefacts help you choreograph wisely." 
