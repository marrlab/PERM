# PERM: PERturbation mapping of Memory genes

Code for **"Memory genes respond strongly to genetic perturbations and have CpG-depleted promoters"**
(Büschel, Nahrstedt, Kuznets-Speck, Marr and Goyal).

Memory genes are genes whose expression is more similar among clonally related cells than among
unrelated cells. This repository identifies them in four lineage-barcoded single-cell datasets with
four methods (CORAL, GEMLI, Power-Seek and ClonePair), builds a consensus, characterises it, and tests
how strongly memory genes respond to genetic perturbation in six paired Perturb-seq screens and
fifteen further screens.

## Layout

```
figures/          figures.py (one function per panel) and the three figure notebooks
src/lib/          shared perturbation-response and matching code
src/memory_genes/ memory-gene calling (clonepair/), consensus, orthologues, GO enrichment
src/screens/      Perturb-seq preparation: pseudobulks and log fold changes
src/gene_properties/  expression-matched comparison on 34 gene features
src/generalisation/   Power-Seek on control cells across fifteen screens
perm_paths.py     all file locations, overridable by environment variables
```

## Installation

```bash
git clone https://github.com/marrlab/PERM.git
cd PERM
conda env create -f environment.yml
conda activate perm
```

Tested on Linux (x86_64). Preparing the Shi screen, Fig. 1g and Supplementary Fig. 8 need 48 to 64 GB
of memory; everything else runs on a workstation.

## Data

No data are stored here. The processed lineage datasets and the Perturb-seq log fold-change matrices
are deposited at Zenodo: [10.5281/zenodo.24590499](https://doi.org/10.5281/zenodo.24590499).

| dataset | system | source |
|---|---|---|
| Michaels et al. | human progenitor T cells | Table S1 |
| Harmange et al. | human melanoma (WM989) | GEO GSE237228 |
| Weinreb et al. | mouse haematopoietic progenitors | GEO GSE140802 |
| Bandler et al. | mouse forebrain | GEO GSE188528 |
| Zhu et al. | CRISPRi, primary human T cells | GEO GSE314342 |
| Weinstock et al. | CRISPR knockout, human CD4⁺ T cells (bulk) | GEO GSE271788 |
| Xu et al. | CRISPRi, A375 melanoma | GEO GSE291147 |
| Frangieh et al. | CRISPR knockout, patient-derived melanoma | SCP1064 / Zenodo 10.5281/zenodo.7041848 |
| Bowness et al. | CRISPRi, mouse haematopoietic progenitors | GEO GSE320250 |
| Shi et al. | in vivo CRISPR knockout, mouse brain | Hugging Face `perturbai/wholebrain_crispr_atlas` |
| fifteen generalisation screens | human cell lines, CRISPRi and CRISPRa | Table S4 |

## Paths

All locations come from environment variables read by `perm_paths.py`:

```bash
cp paths.local.sh.example paths.local.sh   # ignored by git; edit to point at your data
source paths.local.sh
```

The main ones are `PERM_DATA` (processed data), `PERM_MEMGENES_ROOT` (memory-gene lists),
`PERM_OUTPUTS` (results) and `PERM_SCRATCH` (large intermediates); see the example file for all.

## Figures

Each notebook cell draws one panel with one function from `figures/figures.py`. Panels 1a, 2b and
2d are schematics.

| figure | functions |
|---|---|
| 1b–1e | `fig_design_sunburst`, `fig_agreement`, `fig_cross_dataset`, `fig_responsiveness_violin` |
| 1f, 1g | `fig_molecular_features`, `fig_expression_autocorrelation` |
| 2a, 2c, 2e, 2f | `fig_design_sunburst`, `fig_responsiveness_violin` |
| 2g, 2h | `fig_cross_dataset_combo` |
| 2i | `fig_leaveout_curve` |
| 2j | `fig_powerseek_compact` |
| S1 | `fig_scg_threshold`, `fig_ranking_depth`, `fig_topn_sharing`, `fig_ortholog_and_overlap`, `fig_characterisation_matched` |
| S2 | `fig_responsiveness_violin`, `fig_marson_rungs_per_group`, `fig_matching_grid` |
| S3 | `fig_matching_supply`, `fig_robustness_forest`, `fig_donor_replication` |
| S4 | `fig_crossdataset_per_group`, `fig_cross_dataset_sweep`, `fig_cross_dataset_lines`, `fig_leaveout_grid` |
| S5 | `fig_memoryseq_overlap`, `fig_memoryseq_responsiveness` |
| S6 | `fig_lineage_free_responsiveness`, `fig_lineage_free_summary` |
| S7 | `fig_powerseek_summary` |
| S8 | `fig_box`, `fig_box_panel` |

## Pipeline

Only needed to regenerate the intermediate tables; each script documents itself (`--help`).

```bash
# 1. ClonePair memory genes (defaults reproduce the published lists)
python src/memory_genes/clonepair/findingSCGs.py --output_dir $PERM_OUTPUTS/clonepair \
    --data T-lineage $PERM_DATA/datasets/T-lineage_processed.h5ad \
    --data LARRY     $PERM_DATA/datasets/LARRY_processed.h5ad
python src/memory_genes/clonepair/findingSCGs_melanoma.py --output_dir $PERM_OUTPUTS/clonepair/harmange
python src/memory_genes/clonepair/findingSCGs_bandler.py  --output_dir $PERM_OUTPUTS/clonepair/bandler

# 2. Orthologues, consensus, GO
python src/memory_genes/build_ortholog_maps.py
python src/memory_genes/build_consensus.py
python src/memory_genes/go_enrichment.py        # BP by default; --gmt for CC and MF

# 3. Perturb-seq screens
python src/screens/prepare_screen.py --screen zhu   # also frangieh, xu, shi, bowness
python src/screens/prepare_weinstock.py

# 4. Gene features
python src/gene_properties/compute_stats.py
python src/gene_properties/cpg_null_panel.py
```

CORAL, GEMLI and Power-Seek were run with their authors' implementations; Python ports are in
`src/memory_genes/utils.py`. The generalisation analysis runs one job per row of
`src/generalisation/datasets.tsv`.

## Citation

> Büschel, Nahrstedt, Kuznets-Speck, Marr and Goyal. Memory genes respond strongly to genetic
> perturbations and have CpG-depleted promoters. *[journal, year]*.

## License

MIT; see [LICENSE](LICENSE).
