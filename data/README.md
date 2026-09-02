# Dataset

`BEED_Data_clean.csv` is a cleaned copy of the Bangalore EEG Epilepsy Dataset (BEED).

## Contents

- 7,729 rows
- 16 EEG input columns (`X1`–`X16`)
- Target column: `y`
- No missing values
- No duplicate rows

The original UCI record lists 8,000 instances. The included cleaned copy contains 7,729 unique rows. Because the supplied cleaned file does not retain a row identifier or cleaning log, this repository does not claim which exact original rows were removed.

## Labels

| Label | Description |
|---:|---|
| 0 | Healthy subjects |
| 1 | Generalized seizures |
| 2 | Focal seizures |
| 3 | Seizure events with associated activities such as eye blinking, nail biting, or staring |

## Source and license

- Dataset page: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/1134/beed%3A%2Bbangalore%2Beeg%2Bepilepsy%2Bdataset)
- DOI: [10.24432/C5K33B](https://doi.org/10.24432/C5K33B)
- Dataset license: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Citation:

> Najmusseher, & Banu P. K., N. (2024). *BEED: Bangalore EEG Epilepsy Dataset* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K33B

The CC BY 4.0 license applies to the dataset. No additional ownership claim is made over the original data.
