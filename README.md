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

The script performs two-stage estimation for integrated and neutral builders, calculates R-squared statistics, and runs bootstrap resampling (1,000 replications) to compute 95% percentile confidence intervals. The full-sample Stage 1 fit is run from the default start and 20 random starting values, keeping the solution with the lowest objective; bootstrap replications start from the full-sample estimates.

### Replication

- **Data file:** the script reads `aggregated_bids_with_volatility_and_mempool.csv.gz` from the working directory; set `MEVBOOST_DATA` to use a different path.
- **Run log:** each run prints the SHA-256 checksum of the data file, the sample dates, the number of auctions before and after the sample filter, software versions, and optimizer convergence status for each step. `output.txt` is the run log for the results reported in the paper.
- **Randomness:** the bootstrap and the random starting values use a fixed seed (`SEED` in `analysis.py`), and the draws are saved to `bootstrap_draws.csv`.
- **Worked example:** the example block in the paper (block 20,915,921) is selected by `EXAMPLE_BLOCK`.

## Data

The dataset is included as `aggregated_bids_with_volatility_and_mempool.csv.gz` (gzip-compressed CSV, 22 MB). SHA-256 checksums: `5d0441d999838fa6a20166ccc043701193f5da99ca72de8c9b6e3148fcd938e4` for the `.gz` file (the value printed in the run log) and `0c3af446857cef284737e17779e399d334bcc4efc365c048e8ab3962bf0c66b8` for the uncompressed CSV. The file contains approximately 95,000 MEV-Boost auctions from selected days between October and December 2024, including bid data from major builders, CEX price movements, base fees, and mempool statistics. Estimation uses the 68,120 auctions in which Beaverbuild, rsync-builder, Titan, and Flashbots all submit a positive bid. The file is built from relay bid data, on-chain block data, the Flashbots mempool archive, and Tardis Binance ETH-USDC quotes, as described in Section 3 of the paper. 

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
