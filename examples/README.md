MasterMSM examples
==================
Jupyter notebooks covering the fundamentals of the MasterMSM package.
They use either model systems (Brownian dynamics on one and two
dimensional potentials) or molecular dynamics simulations of simple
peptides.

Contents
--------
| Folder | Notebooks | Content |
|--------|-----------|---------|
| `alanine_dipeptide` | `ala_dipeptide.ipynb` | Complete workflow: Ramachandran discretization, MSM, convergence and Chapman-Kolmogorov tests, mode autocorrelation functions, Taylor rate matrix |
| | `ala_dipeptide_maxlike.ipynb` | Rate matrix with the maximum likelihood propagator-based (MLPB) method |
| `alanine_pentapeptide` | `ala_pentapeptide.ipynb` | MSM of a longer peptide from Ramachandran states |
| | `ala_pentapeptide_tica.ipynb` | Dimensionality reduction with TICA |
| | `discretization_comparison.ipynb` | Comparison of discretization protocols (Ramachandran, k-means, HDBSCAN on PCA and TICA) |
| `bistable_potential` | `1D_smFS_MSM.ipynb`, `2D_smFS_MSM.ipynb` | Brownian dynamics on one and two dimensional two-state surfaces |
| `schutte_potential` | `schute_mastermsm.ipynb` | Brownian dynamics on the one dimensional Schütte potential |

Data
----
The MD data for the peptides are hosted at [OSF](https://osf.io/a2vc7)
and are downloaded by the notebooks themselves. The Brownian dynamics
data for the bistable and Schütte potentials are fetched with

    python download_data.py

run from this folder.

Some notebooks use additional packages for plotting (`seaborn`) or
clustering (`hdbscan`).
