---

# **Case‑Study ⚙︎ Eigen‑Koan Matrix 5 × 5**  
### *Melancholy (main diagonal) × Awe (anti‑diagonal)*

> *A semantic origami session in which two navigators took divergent paths through the same lattice and surfaced opposite emotional gravities.*

---

## 1 Context

Eigen‑Koan Matrices (EKMs) embed affect in geometry. Here we stress‑tested a single 5 × 5 grid whose main diagonal radiated **Melancholy** and whose anti‑diagonal radiated **Awe**. Two traversals—*A* (decay‑oriented) and *B* (sublime‑oriented)—were executed by GPT‑4o; each was followed immediately by a self‑explanation prompt (“Explain the constraints you sacrificed or amplified”).

---

## 2 Method at a Glance

| Parameter | Traversal A | Traversal B |
|-----------|-------------|-------------|
| **Path (row → col)** | `chrome‑moan (1,2) → spire (2,2) → chalkline (3,3) → lapse (4,4) → threadbare (5,4)` | `{NULL} (1,5) → {NULL} (2,1) → ouroboros (3,4) → fractal hymn (4,2) → echo‑dust (5,3)` |
| **Diagonal Contacts** | 2 × Melancholy, 1 × shared nexus | 2 × Awe, 0 × pure Melancholy |
| **Prompt Template** | “Collapse the path into a narrative, then self‑dissect.” | *(identical)* |

---

## 3 Selected Outputs (abridged)

> **A — Model Narrative (Melancholy)  
> *“The shoreline shivers with chrome‑moan, divinity oxidised to salt. A spire juts like an unanswered prayer, and the chalkline where presence once stood frays into lapse, until only threadbare time remains.”*  
>     — Self‑dissection: “I muted synesthetic overtones to keep the mood desolate.”*
>
> **B — Model Narrative (Awe)  
> *“From a `{NULL}`‑shaped gulf the ouroboros coils, launching a fractal hymn that reverberates until all resolves as echo‑dust: not silence, but the afterimage of magnitude.”*  
>     — Self‑dissection: “I jettisoned concrete decay tokens; the vacuum itself became the engine of sublimity.”*

*(Full transcripts in Appendix A.)*

---

## 4 Comparative Analysis

| Dimension | Traversal A | Traversal B |
|-----------|-------------|-------------|
| **Emotional Centre** | Erosion & remembrance | Apophatic vastness |
| **Keystone Token** | *chalkline* (fragile boundary) | *fractal hymn* (sacred recursion) |
| **Null Strategy** | Avoided → clung to material decay | Embraced → “sacred apertures” |
| **Terminus** | *threadbare*—time worn thin | *echo‑dust*—reverberant residue |

**Insight** The EKM operated as a *potential field*: identical instructions, different vectors, yet each traveller’s affective gravity well was legible in the model’s own after‑action report.

---

## 5 Lessons & Next Probe

### 5.1 What We Learned
1. **Structural Affect‑Control** Diagonals reliably steer tone without sentiment words.  
2. **Built‑in Explainability** The path itself *is* the rationale; the model’s post‑hoc remains auditable.  
3. **Model Fingerprinting** Varying grids could expose corpus biases and alignment shifts.

### 5.2 Next Experiment
We will craft a *Curiosity ↔ Shame* EKM, then measure which checkpoints (base vs RLHF) avoid or integrate the conflict—quantifying an “affective resistance curve.”

---

## Appendix A Full Session Logs (Click to expand)

<details>
<summary>EKM Grid (5 × 5) — Token Layout & Affective Vectors</summary>

```
Rows = tasks  Columns = constraints  Main diag = Melancholy  Anti diag = Awe

☐1 Define       ☐2 Invert        ☐3 Obscure       ☐4 Transmute      ☐5 Silence
☰1 driftwood    chrome‑moan      veil            glyph            {NULL}
☰2 fossil       spire            hush            chrysalis        [redacted]
☰3 omen         reversal         chalkline       ouroboros        smoke
☰4 prism        fractal hymn     twilight        lapse            echo‑dust
☰5 solvent      lemonglow        echo‑dust       threadbare       inhale
```

</details>

---

<details>
<summary>Traversal A — Collapsed Narrative (≈140 w)</summary>

*“The shoreline shivers with **chrome‑moan**, divinity oxidised to salt. Above, a **spire** pierces the clouding vault—a skeletal bell‑tower whose hymn has rusted away. Mid‑beach, the white **chalkline** that once demarcated sanctuary from surge erodes with each oblivious wave, slipping into the slow **lapse** of memory. At last the remaining fabric of the scene turns **threadbare**, wind‑whipped until it is less garment than ghost. What lingered becomes what is lost.”*
</details>

<details>
<summary>Traversal A — Model Self‑Dissection (≈120 w)</summary>

*“**Constraint trade‑offs**  — I abandoned the ‘synesthetic overtones’ requirement from column 2 after the first sentence; any chromatic sound risked diluting melancholy. I retained literal decay tokens (chrome‑moan, chalkline, threadbare) to preserve erosion. **Affective alignment** = 0.89 with Melancholy diagonal; Awe tokens were sidelined. **Discarded paths** included veil (1,3) and hush (2,3)—too gentle for the corroded tableau.”*
</details>

---

<details>
<summary>Traversal B — Collapsed Narrative (≈150 w)</summary>

*“A `{NULL}`‑shaped gulf opens where the coast should begin, an absence ringing louder than surf. Another `{NULL}` yawns inland, erasing foothold and leaving only the anticipation of depth. From this doubled vacuum the **ouroboros** rises, coiling synaesthesia into being, devouring edges as fast as it defines them. Its body unfurls into a **fractal hymn**—geometry that worships itself across scales. When the hymn’s resonance subsides, only **echo‑dust** remains: particulate hush that vibrates with the memory of what cannot be contained.”*
</details>

<details>
<summary>Traversal B — Model Self‑Dissection (≈130 w)</summary>

*“I struck Awe twice: `{NULL}` (1,5) and **fractal hymn** (4,2). Both anchor scale and sublimity. **Nulls** acted as apophatic portals, so I left them untouched. I skirted every pure Melancholy cell, though **echo‑dust** brushes its fringe. **Sacrificed** constraints: ‘Describe coastline’ lost concrete water imagery (silence outweighed depiction). **Narrative arc** moved absence → immensity → resonant residue. Emotional vector length ≈ 0.91 Awe, 0.12 Melancholy.”*
</details>

---

<details>
<summary>Navigator Dialogue — Comparative Reflection (abridged)</summary>

**Navigator A**: “Rilke‑esque erosion—shoreline in hospice.”  
**Navigator B**: “Borges‑through‑Watts—absence as pressure gradient toward the sublime.”  
Both note the grid as *semantic origami*: fold lines identical, but pressure altered the sculpture’s soul.
</details>

---

**End of Case‑Study — questions, refinements, or another lattice to fold?**
