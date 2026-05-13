# ATLAS Paper

LaTeX source for *ATLAS: A Federated Multi-Task Framework for Privacy-Preserving Threat-Intelligence Estimation Across Regional Fusion Centers* — the paper draft based on the fusion-centers FL work in this repository.

## Format

- **Class:** `IEEEtran` conference style (matches the HERMES paper template).
- **Bibliography:** inline `thebibliography` block (no external `.bib` file needed). Swap to `\bibliography{biblio}` + a `biblio.bib` if you prefer BibTeX management.
- **Figures:** placeholder paths under `Figures/` — these need to be generated from actual experiment runs before the paper compiles a final PDF. The plot scripts in `Analysis/CommunitiesCrime/` produce the right artifacts; see [`DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md` §4](../DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md) for the exact commands.

## Section structure (mirrors HERMES)

1. **Title + Abstract + Keywords**
2. **I. Introduction** — fusion-center mission, three coupled challenges (geographic non-IID, multi-task heads, statutorily sensitive features), ATLAS contribution, paper roadmap
3. **II. Related Works** — Table I comparison across FL frameworks, then 5 paragraphs on the threads (FL basics, tabular FL, multi-task FL, fairness-aware FL, cross-jurisdictional systems)
4. **III. ATLAS Engineering Methodology**
   - A. Layer 1: Privacy-Preserving Data Partitioning Layer
   - B. Layer 2: Multi-Task FUSION-MLP Layer
   - C. Layer 3: Federated Coordination Layer + Algorithm 1
   - Complexity Analysis
5. **IV. ATLAS Experiments and Results Discussion**
   - A. Experiment Setup (6-experiment matrix table, sensitive-features ablation, multi-node validation)
   - B. Evaluation Metrics (table of dependent variables, Tier 1 / Tier 2)
   - C. Evaluation Results (Group A = learning architecture, Group B = federation strategy + scaling, sensitive-features ablation) with **Observation 1–5** callouts
6. **V. Conclusion and Future Work**
7. **References**

## Placeholder content

The result tables and figure references use **illustrative numbers** consistent with the design's Phase B DoD threshold (macro-F1 > 0.55 for the centralized baseline). Once the actual experiments complete:

- Replace numbers in `tab:learning_arch_results`, `tab:federation_strategy`, `tab:scaling_results`, `tab:bias_ablation` with the macro-F1 values from each `<run_dir>/<timestamp>_evaluation.log`.
- Drop the actual generated PNG files into `Figures/`:
  - `headline_convergence.png` — from `plot_centralized_vs_federated`
  - `per_client_distribution_geo_n5.png` — from `plot_per_client_distribution`
  - `proximal_evolution.png` — from `plot_centralized_vs_federated --metric proximal_contribution`
  - `atlas_architecture.png` — three-layer architecture diagram (still to draw)
  - `overview_atlas.png` — splash figure (still to draw)
- Remove the `(Illustrative numbers.)` parenthetical from each table caption.

## Compile

```bash
# Linux / macOS / WSL
cd paper
pdflatex main.tex && pdflatex main.tex   # twice for refs
```

```powershell
# Windows
cd paper
pdflatex main.tex; pdflatex main.tex
```

Compilation will warn about missing `Figures/*.png` until the placeholder figures are populated. The text and references compile cleanly.
