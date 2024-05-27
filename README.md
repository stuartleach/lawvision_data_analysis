[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![Actions Status](https://github.com/stuartleach/lawvision_data_analysis/workflows/Ruff-Lint/badge.svg)](https://github.com/stuartleach/lawvision_data_analysis/actions) 

### An Exploratory Analysis of Pretrial Data in New York State

"Bail" is the fee a defendant can pay to be released from custody pending trial. The purpose of bail is to ensure that the defendant appears in court for their trial. If the defendant fails to appear in court, the bail is forfeited. Otherwise, the defendant recoups the expense.

The bail system is controversial because it can lead to the pretrial detention of individuals who cannot afford to pay bail.

In New York City, this often means a stint at Rikers Island, a notorious jail complex that has been the subject of numerous investigations and lawsuits. 

[CPL 510.10](https://www.nysenate.gov/legislation/laws/CPL/510.10) outlines what judges _**can**_ and _**cannot**_ consider when setting bail.

### _The purpose of this project is to understand how closely judges in New York State adhere to the law, as outlined in CPL 510.10, when setting bail._
### Dataset:
1. [New York Pretrial Release Data](https://ww2.nycourts.gov/pretrial-release-data-33136)
2. Census Average Median Income for New York State counties (TODO: link for reference)

### Method:
1. Create a "New York Judge" profile by analyzing the bail decisions of judges in New York State and using regression models such as Gradient Boosting and Random Forest to assign importance values to various features that might be relevant to the case.
2. Create a similar profile for each county; i.e., find what's more important to a Kings County judge than a Richmond County judge when compared to the average New York State judge.
3. Assign profiles to every judge in the state, comparing them to judges in their county and state. 

### Questions we will explore:
1. How closely to judges follow the law when setting bail? Are there factors that have an outsized influence on the amount of bail set?
2. What factors have a high importance value that _shouldn't_, e.g. race, gender, median income of the county?
3. What factors have a higher importance value to some judges than others? One judge, for example, might deem a defendant's age as more significant than prior criminal convictions.

**This is not, and will never be,** a tool to be used by counsel. It can't teach a lawyer what to emphasize on their client's behalf. This model will never be as effective as a reasonably perceptive defense attorney. This project is only to understand the extent that judges in New York State abide by the rules laid out in CPL 510.10.



