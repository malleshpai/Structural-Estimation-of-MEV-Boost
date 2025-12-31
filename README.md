# Structural Estimation of MEV-Boost Auctions

This repository contains the code for the paper "Structural Estimation of MEV-Boost Auctions" by Tivas Gupta, Mallesh M. Pai, Max Resnick, and Xun Tang.

## Description

This project implements a structural econometric model to estimate the distribution of builder values in MEV-Boost auctions on Ethereum. The model uses a two-stage non-linear least squares procedure to decompose builder values into common and idiosyncratic components, separately identifying type-specific factors from builder-specific private information.

## Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Running the Estimation

```bash
python analysis.py
```

The script performs two-stage estimation for integrated and neutral builders, calculates R-squared statistics, and runs bootstrap resampling (500 iterations) to compute 95% confidence intervals.

## Data

The dataset (`aggregated_bids_with_volatility_and_mempool.csv`) is too large for GitHub. The data contains approximately 95,000 MEV-Boost auctions from selected days between October and December 2024, including bid data from major builders, CEX price movements, base fees, and mempool statistics. 

**Data is available from the authors upon request.**

## Citation

If you use this code, please cite:

```bibtex
@article{gupta2025structural,
  title={Structural Estimation of MEV-Boost Auctions},
  author={Gupta, Tivas and Pai, Mallesh M. and Resnick, Max and Tang, Xun},
  journal={Working Paper},
  year={2025}
}
```
