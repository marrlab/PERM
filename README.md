# PERM: PERturbation mapping of Memory genes

Code for **"Memory genes respond strongly to genetic perturbations and have CpG-depleted promoters"**
(Büschel, Nahrstedt, Kuznets-Speck, Marr and Goyal).

Memory genes are genes whose expression is more similar among clonally related cells than among
unrelated cells. This repository identifies them in four lineage-barcoded single-cell datasets with
four methods (CORAL, GEMLI, Power-Seek and our method, ClonePair), builds a consensus across methods,
characterises the consensus genes, and tests how strongly they respond to genetic perturbation in six
Perturb-seq screens paired by cell type, plus fifteen further screens for generalisation. It
reproduces every figure panel and supplementary table of the paper from the processed data.

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data](#data)
- [Configuring paths](#configuring-paths)
- [Reproducing the figures](#reproducing-the-figures)
- [Running the pipeline](#running-the-pipeline)
- [Names in the code](#names-in-the-code)
- [Citation](#citation)
- [License](#license)

## Repository layout

```
PERM/
├── environment.yml           conda environment (pinned versions)
├── perm_paths.py             every filesystem location the code uses, overridable by environment variables
├── paths.local.sh.example    template for your own data locations
├── figures/
│   ├── figures.py            one function per figure panel
│   ├── figure1.ipynb         Figure 1
│   ├── figure2.ipynb         Figure 2
│   └── suppl_figures.ipynb   Supplementary Figures
└── src/
    ├── lib/                  shared perturbation-response and matching code
    ├── memory_genes/         memory-gene calling, consensus, orthologues, GO enrichment
    │   └── clonepair/        ClonePair, our memory-gene caller, and its threshold sweep
    ├── screens/              Perturb-seq preparation: pseudobulk, log fold changes, per-donor and per-condition splits
    ├── gene_properties/      expression-matched comparison of memory genes on 34 gene features
    └── generalisation/       Power-Seek on control cells and response testing in fifteen further screens
```

## Installation

The code was run on Linux (x86_64) with the versions pinned in `environment.yml`.

```bash
git clone https://github.com/vsarsam/PERM.git
cd PERM
conda env create -f environment.yml
conda activate perm
```

Several steps hold a large screen in memory: preparing the Shi screen, Fig. 1g and Supplementary
Fig. 8 need about 48 to 64 GB. Everything else runs on a workstation.

## Data

No data are stored in this repository. The processed data needed to reproduce the figures, including
every intermediate table the figures read, are deposited at **Zenodo: [DOI to be added]**. Download
the archive and point `PERM_DATA` at it (see [Configuring paths](#configuring-paths)).

The original data are public:

| dataset | system | source |
|---|---|---|
| Michaels et al. | human haematopoietic and progenitor T cells, transcribed barcodes | Table S1 |
| Harmange et al. | human melanoma (WM989), scMemorySeq | GEO GSE237228 |
| Weinreb et al. | mouse haematopoietic progenitors, LARRY | GEO GSE140802 |
| Bandler et al. | mouse forebrain, STICR | GEO GSE188528 |
| Zhu et al. | genome-scale CRISPRi, primary human T cells | GEO GSE314342 |
| Weinstock et al. | arrayed CRISPR knockout, primary human CD4⁺ T cells (bulk) | GEO GSE271788 |
| Xu et al. | CRISPRi, A375 melanoma (PerturbFate) | GEO GSE291147 |
| Frangieh et al. | CRISPR knockout, patient-derived melanoma (Perturb-CITE-seq) | SCP1064, via scPerturb (Zenodo 10.5281/zenodo.7041848) |
| Bowness et al. | CRISPRi, primary mouse haematopoietic progenitors | GEO GSE320250 |
| Shi et al. | in vivo CRISPR knockout, mouse brain | Hugging Face `perturbai/wholebrain_crispr_atlas` |
| fifteen generalisation screens | eight human cell lines, CRISPRi and CRISPRa | Table S4 |

Gene annotations come from published resources: mRNA half-lives (Agarwal and Kelley), transcriptional
bursting (Boe et al.), MSigDB v2026.2 Gene Ontology collections, NCBI gene orthologues and gene
information, gnomAD v4.1, STRING v12, InterPro, GENCODE (human release 47, mouse release M36) and the
UCSC CpG-island and phastCons tracks. Versions and download dates are given in the Methods.

The four lineage datasets are deposited as processed objects (quality-controlled, unassigned cells
removed, log1p counts per 10,000). Michaels and Weinreb were received already quality-controlled
from the original authors; Harmange and Bandler were assembled from the deposits listed above. The
tables behind Supplementary Fig. 8 panels c, d, f and g, and the per-gene feature matrices behind
the gene-feature screen, are deposited as computed.

## Configuring paths

All code reads its locations from `perm_paths.py`. Every location can be overridden by an environment
variable, so nothing in the repository needs editing:

```bash
cp paths.local.sh.example paths.local.sh     # paths.local.sh is ignored by git
# edit paths.local.sh to point at your copy of the data, then:
source paths.local.sh
```

| variable | holds |
|---|---|
| `PERM_DATA` | processed datasets, log-fold-change tables, annotations (the Zenodo archive) |
| `PERM_RAW` | raw screen downloads, referenced by `src/generalisation/datasets.tsv` |
| `PERM_MEMGENES_ROOT` | per-method memory-gene lists and the consensus and GO tables |
| `PERM_OUTPUTS` | result tables written by the scripts |
| `PERM_SCRATCH` | large intermediates (pseudobulks, generalisation fold changes) |
| `PERM_FEATURES` | per-gene feature matrices for the gene-feature screen |
| `PERM_BOX` | tables for Supplementary Fig. 8 (default `$PERM_DATA/box_figure`) |

Variables left unset fall back to folders inside the repository.

## Reproducing the figures

Each notebook cell draws one panel by calling one function in `figures/figures.py`, and writes the
panel to `figures/panels/`. Start Jupyter after `source paths.local.sh` so the kernel inherits the
paths. Panels 1a, 2b and 2d are schematics and have no code.

**Figure 1** (`figures/figure1.ipynb`)

| panel | content | function |
|---|---|---|
| 1b | datasets, tissues and groups | `fig_design_sunburst` |
| 1c | agreement between methods | `fig_agreement` |
| 1d | memory genes shared across datasets | `fig_cross_dataset` |
| 1e | expression of consensus memory genes | `characterisation_table`, `fig_responsiveness_violin` |
| 1f | mRNA half-life, bursting and promoter CpG content | `molecular_feature_table`, `fig_molecular_features` |
| 1g | expression autocorrelation across cell divisions | `fig_expression_autocorrelation` |

**Figure 2** (`figures/figure2.ipynb`)

| panel | content | function |
|---|---|---|
| 2a | screens paired with each lineage dataset | `fig_design_sunburst` |
| 2c | impact of memory-gene perturbations | `fig_responsiveness_violin` |
| 2e, 2f | response of memory genes per method and per matching covariate | `fig_responsiveness_violin` |
| 2g, 2h | response across the six pairings | `xd_sweep`, `xd_consensus_sweep`, `fig_cross_dataset_combo` |
| 2i | robustness to removing the strongest perturbations | `leaveout_curve_data`, `fig_leaveout_curve` |
| 2j | lineage-free memory genes across fifteen screens | `powerseek_responsiveness`, `fig_powerseek_compact` |

The last cell of `figure1.ipynb` and `figure2.ipynb` writes the Source Data workbook for that figure.

**Supplementary Figures** (`figures/suppl_figures.ipynb`)

| figure | content | function |
|---|---|---|
| 1a, 1b | ClonePair threshold | `load_scg_sweep`, `fig_scg_threshold` |
| 1c | list length per method | `ranking_depth_table`, `fig_ranking_depth` |
| 1d | cross-dataset sharing across cutoffs | `fig_topn_sharing` |
| 1e, 1f | orthologue mapping and pairwise overlap | `fig_ortholog_and_overlap` |
| 1g | detection rate, protein abundance and gene length, expression-matched | `fig_characterisation_matched` |
| 2a | impact of memory-gene perturbations, per experimental group | `pergroup_memory_sets`, `fig_responsiveness_violin` |
| 2b | response per matching covariate, per Zhu group | `fig_marson_rungs_per_group` |
| 2c | robustness to the background-matching scheme | `fig_matching_grid` |
| 3a | supply of expression-matched controls | `matching_supply_table`, `fig_matching_supply` |
| 3b | robustness to analysis choices | `robustness_tables`, `fig_robustness_forest` |
| 3c | replication across donors | `donor_responsiveness`, `fig_donor_replication` |
| 4a | response per screen and lineage group | `fig_crossdataset_per_group` |
| 4b | response across cutoffs | `xd_sweep`, `fig_cross_dataset_sweep` |
| 4c | response by consensus tier | `fig_cross_dataset_lines` |
| 4d | robustness to removing perturbations, per method | `fig_leaveout_grid` |
| 5a, 5b | comparison with bulk MemorySeq | `fig_memoryseq_overlap`, `fig_memoryseq_responsiveness` |
| 6a, 6b | memory genes found without lineage barcodes | `fig_lineage_free_responsiveness`, `fig_lineage_free_summary` |
| 7 | Power-Seek on control cells, every screen | `powerseek_responsiveness`, `fig_powerseek_summary` |
| 8a–8g | structure of perturbation responses | `box_compute`, `fig_box`, `fig_box_panel`, `box_stats` |

## Running the pipeline

The figures read deposited intermediate tables, so the steps below are only needed to regenerate
those tables. Every script documents its inputs, outputs and options (`python <script> --help`).

**1. Memory genes.** ClonePair is run on each lineage dataset; the defaults reproduce the published
lists (clones of 2 to 8 cells, concordance threshold 0.80, seed 0):

```bash
python src/memory_genes/clonepair/findingSCGs.py --output_dir $PERM_OUTPUTS/clonepair \
    --data T-lineage $PERM_DATA/datasets/T-lineage_processed.h5ad \
    --data LARRY     $PERM_DATA/datasets/LARRY_processed.h5ad
python src/memory_genes/clonepair/findingSCGs_melanoma.py --output_dir $PERM_OUTPUTS/clonepair/harmange
python src/memory_genes/clonepair/findingSCGs_bandler.py  --output_dir $PERM_OUTPUTS/clonepair/bandler
```

CORAL, GEMLI and Power-Seek were run with their authors' released implementations (Methods);
`src/memory_genes/utils.py` contains the Python ports of all three, which the lineage-free analyses
use. The per-method lists for the four datasets are part of the deposit.

**2. Consensus and annotation.**

```bash
python src/memory_genes/build_ortholog_maps.py    # mouse-to-human orthologues (NCBI)
python src/memory_genes/build_consensus.py        # consensus memory genes, exactly as the figures define them
python src/memory_genes/go_enrichment.py --help   # GO over-representation; see the script for the three ontologies
```

**3. Perturb-seq screens.** One script prepares five screens; Weinstock, a bulk screen, has its own:

```bash
python src/screens/prepare_screen.py --screen zhu        # also: frangieh, xu, shi, bowness
python src/screens/prepare_weinstock.py
```

`prepare_velten.py` and `build_shi_pseudobulk.py` build the inputs for the Bowness and Shi screens,
and `marson_per_group.py`, `marson_per_donor.py` and `marson_per_cell.py` split the Zhu screen by
activation condition, donor and cell for the supplementary analyses.

**4. Gene features.**

```bash
python src/gene_properties/compute_stats.py    # 34 features, expression-matched (Supplementary Data)
python src/gene_properties/cpg_null_panel.py   # promoter CpG content against an expression-matched null
```

**5. Generalisation.** Fifteen screens, one job per row of `src/generalisation/datasets.tsv`:
`qc_filter_controls.py` (control-cell quality control), `make_tasks.py`, `powerseek_control.py`
(Power-Seek on control cells, one group per job), `build_pseudobulk.py` and
`build_double_pseudobulk.py` (pseudobulks, single and paired perturbations) and `write_lfc.py`
(log fold changes).

## Names in the code

Internal names in the code predate the names used in the paper:

| in the code | in the paper |
|---|---|
| `SCG`, `findingSCGs*.py`, `scg_memory_genes_*` | ClonePair |
| `T-lineage` | Michaels et al. |
| `LARRY` | Weinreb et al. |
| `Bandler`, `BandlerBrain` | Bandler et al. |
| `Harmange`, `Melanoma_Harmange` | Harmange et al. |
| `marson` | Zhu et al. screen |
| `velten` | Bowness et al. screen |
| `mold`, `Mold_brain` | Shi et al. screen |
| `melanoma`, `perturbfate` (screen files) | Xu et al. screen |

## Citation

If you use this code, please cite:

> Büschel, Nahrstedt, Kuznets-Speck, Marr and Goyal. Memory genes have CpG-depleted promoters and
> respond strongly to genetic perturbations. *[journal and year to be added]*.

## License

Released under the MIT License; see [LICENSE](LICENSE).
