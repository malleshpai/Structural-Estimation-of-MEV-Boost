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

The script performs two-stage estimation for integrated and neutral builders, calculates R-squared statistics, and runs bootstrap resampling (1,000 replications) to compute 95% percentile confidence intervals.

To check the Stage 1 estimates against 20 random starting values, run:

```bash
python analysis.py --multistart
```

### Replication

- **Data file:** set `MEVBOOST_DATA` to the path of the estimation file if it is not in the working directory.
- **Run log:** each run prints the SHA-256 checksum of the data file, the sample dates, the number of auctions before and after the sample filter, software versions, and optimizer convergence status for each step. `output.txt` is the run log for the results reported in the paper.
- **Randomness:** the bootstrap uses a fixed seed (`SEED` in `analysis.py`), and the draws are saved to `bootstrap_draws.csv`.
- **Worked example:** the example block in the paper (block 20,915,921) is selected by `EXAMPLE_BLOCK`.

## Data

The dataset (`aggregated_bids_with_volatility_and_mempool.csv`) is too large for GitHub. The data contains approximately 95,000 MEV-Boost auctions from selected days between October and December 2024, including bid data from major builders, CEX price movements, base fees, and mempool statistics. Estimation uses the 68,120 auctions in which Beaverbuild, rsync-builder, Titan, and Flashbots all submit a positive bid. The file is built from relay bid data, on-chain block data, the Flashbots mempool archive, and Tardis Binance ETH-USDC quotes, as described in Section 3 of the paper. 

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
