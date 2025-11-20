# GRB-Light-Curve-Reconstruction-2
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![arXiv](https://img.shields.io/badge/arXiv-2506.23681-b31b1b.svg)](https://arxiv.org/abs/2506.23681)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

Reconstructs 521 Swift GRB light curves using 7 ML/statistical models (e.g., QSS, CNN-BiLSTM, DGP). The Quartic Smoothing Spline (QSS) reduces parameter uncertainty by ~43-48%, enhancing GRBs as cosmological standard candles. Includes preprocessing, training, and uncertainty pipelines.


## Abstract
Mitigating data gaps in Gamma-ray bursts (GRBs) light curves (LCs) is crucial for cosmological research, enhancing the precision of parameters, assuming perfect satellite conditions for complete LC coverage with no gaps. This analysis improves the
applicability of the two-dimensional Dainotti relation, which connects the rest-frame end time of the plateau emission (Ta) and its
luminosity (La), derived from the fluxes (Fa). The study expands on a previous 521 GRB sample by incorporating seven models:
Deep Gaussian Process (DGP), Temporal Convolutional Network (TCN), Hybrid CNN with Bidirectional Long Short-Term Memory (CNN-BiLSTM), Bayesian Neural Network (BNN), Polynomial Curve Fitting, Isotonic Regression, and Quartic Smoothing
Spline (QSS). Results indicate that QSS significantly reduces uncertainty across parameters—43.5% for log Ta, 43.2% for log Fa,
and 48.3% for α, outperforming the other models where α denotes the slope post-plateau based on Willingale’s 2007 functional
form. The Polynomial Curve Fitting model demonstrates moderate uncertainty reduction across parameters, while CNN-BiLSTM
has the lowest outlier rate for α at 0.77%. These models broaden the application of machine-learning techniques in GRB LC analysis, enhancing uncertainty estimation and parameter recovery, and complement traditional methods like the Attention U-Net and
Multilayer Perceptron (MLP). These advancements highlight the potential of GRBs as cosmological probes, supporting their role in
theoretical model discrimination via LC parameters, serving as standard candles, and facilitating GRB redshift predictions through
advanced machine-learning approaches.

## 🛠️ Installation

```bash
git clone [https://github.com/Krishnanjan-Sil/GRB-Light-Curve-Reconstruction-2.git](https://github.com/Krishnanjan-Sil/GRB-Light-Curve-Reconstruction-2.git)
cd GRB-Light-Curve-Reconstruction-2
pip install -r requirements.txt
```

## 📚 Citation

If you use this code or dataset in your research, please cite the following paper:

```bibtex
@misc{kaushal2025multimodelframeworkreconstructinggammaray,
      title={Multi-Model Framework for Reconstructing Gamma-Ray Burst Light Curves}, 
      author={A. Kaushal and A. Manchanda and M. G. Dainotti and K. Gupta and Z. Nogala and A. Madhan and S. Naqi and Ritik Kumar and V. Oad and N. Indoriya and Krishnanjan Sil and D. H. Hartmann and M. Bogdan and A. Pollo and JX. Prochaska and N. Fraija},
      year={2025},
      eprint={2506.23681},
      archivePrefix={arXiv},
      primaryClass={astro-ph.HE},
      url={[https://arxiv.org/abs/2506.23681](https://arxiv.org/abs/2506.23681)}, 
}
