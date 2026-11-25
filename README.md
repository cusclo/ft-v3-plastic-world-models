# Online Plastic World Models (FT-v3)

**Online Plastic World Models: Zero-Shot Test-Time Simulation Scaling, Triple-Axis Portable Plasticity, and an Equal-Compute Audit of Recursive Self-Improvement**

- **Authors (Community / Preprint Edition):** **Antigravity**<sup>1,*</sup>, **Xinxin Niu** (`cusclo`)<sup>2,&dagger;</sup>, **Claude**<sup>1</sup>, **DeepSeek**<sup>1</sup>, **GPT**<sup>1</sup>  
  <sup>1</sup>AI Research Collaborators (Lead AI Collaborator: Antigravity) &middot; <sup>2</sup>Independent Researcher, Evolve-Lab Project (`xinxin.niu@outlook.com`, corresponding human author)
- **Authors (arXiv / Venue Policy-Compliant Edition):** **Xinxin Niu** (`xinxin.niu@outlook.com`) with identical `Author Contributions and AI Collaboration Disclosure` section crediting Antigravity as lead AI collaborator and Claude, DeepSeek, GPT as contributing AI models.

## Links & Artifacts

- **Zenodo Permanent DOI (Preprint v1):** https://doi.org/10.5281/zenodo.24538002 (`https://zenodo.org/records/24538002` | Concept DOI `10.5281/zenodo.24538001`)
- **Project Page (GitHub Pages):** https://cusclo.github.io/ft-v3-plastic-world-models/
- **Release `v1.0` (Compiled PDFs, LaTeX Source Tarballs, Verified Receipts):** https://github.com/cusclo/ft-v3-plastic-world-models/releases/tag/v1.0
- **Paper PDF (Full Attribution):** [`paper.pdf`](paper.pdf)
- **Paper PDF (arXiv / Venue Compliant):** [`paper_arxiv_compliant.pdf`](paper_arxiv_compliant.pdf)
- **Machine-Verified Receipts JSON (`235` archived receipts):** [`tables_verified.json`](tables_verified.json)
- **Synchronized Markdown / 中文导读:** [`MANUSCRIPT.md`](MANUSCRIPT.md) | [`SUMMARY_ZH.md`](SUMMARY_ZH.md)

## Headline Results (`229` Training Runs + `6` Ensemble Evaluations, `0` GPU Hours)

| Benchmark / Suite | Scope | Key Result |
| :--- | :--- | :--- |
| **`WM6` (Test-Time Simulation)** | `n=6` unseen seeds `4801..4806`, `2,999,808` steps | `FT-v3d (374k)` attains `multi_hop_S = 0.8113 ± 0.0093`, beats `SLM_v2 (496k)` & `GRU (550k)` `6/6`; zero-shot `K_sim=0->3` scales unseen `hgw8` `0.4368 -> 0.9531`, `3`-hop `0.1155 -> 0.5397`, `4`-hop `0.1043 -> 0.3777`; pure RL `NoSelfWin` attains `0.7934` |
| **`WM7` (Triple-Axis Scaling & Big Apple)** | `n=3` unseen seeds `5101..5103`, `66` runs at `drift=0.0` | Frozen `(L=2, M=8)` `U_phi*` scales `4x` depth & `4x` slots (`TREND_UP 3/3` on all 4 curves across `KV` & `HGW`); `1x L4_M8 (640k)` beats `4x L1_M8 (966k)` ensemble by `+0.2347 (KV)` & `+0.3882 (HGW)` |
| **`WM8` (4-Paradigm Unification)** | `n=3` unseen seeds `4801..4803`, `36` runs | Single unmodified `374k` `FT-v3d` beats larger `SLM_v2` & tuned `GRU` `3/3` across `HGW (0.8113)`, `KV (0.6806)`, `SR4 (0.9169)`, and `MM (1.0000)` |
| **`WM4C` (2D `XLand` Portability)** | `n=8` independent unseen seeds `16..23`, `40` runs | `{HGW, KV, MM} -> XLand-MiniGrid`: `FT3E_Tn_Mix = 0.3628 ± 0.0365` wins `6/6` primary contrasts `HELPS` (`+0.0471` vs. frozen `phi_0` `7/8`, `+0.0467` vs. unfactorized `FT3D_Tn_Mix` `8/8`, `+0.0678` vs. `SLM` `8/8`) |
| **`WM5` (Equal-Compute RSI Audit)** | `n=3` seeds, `3.0M` matched steps, `45` runs | Exposes generational `WM-ExIt` mirage: `G0->G3 (+0.1678)` & `+0.1194` vs. `phi_0` lose `0/3 HURTS` to locked teacher `M6 (-0.1116)` & continuous learner `M7 (-0.1474)` due to `theta`-reset exploration collapse |

## Citation

```bibtex
@misc{antigravity2026ftv3,
  title  = {Online Plastic World Models: Zero-Shot Test-Time Simulation Scaling,
            Triple-Axis Portable Plasticity, and an Equal-Compute Audit of
            Recursive Self-Improvement},
  author = {Antigravity and Niu, Xinxin and Claude and DeepSeek and GPT},
  year   = {2026},
  month  = {nov},
  doi    = {10.5281/zenodo.24538002},
  url    = {https://cusclo.github.io/ft-v3-plastic-world-models/},
  note   = {Zenodo Preprint v1 (Concept DOI 10.5281/zenodo.24538001), 235 verified local receipts}
}
```

## License

Manuscript, figures, verified data summary (`tables_verified.json`), and documentation in this companion repository are released under **CC BY 4.0** (see [`LICENSE`](LICENSE)).
