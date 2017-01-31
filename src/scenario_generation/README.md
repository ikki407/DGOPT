# Scenario Generation

## Files

 - `scenario_generation.py`: Scenario generation with K-means

 - `scenario_generation_kde.py`: Scenario generation by using kernel density estimation. This script estimates the probability density functions (PDF) of demand & price, wind speed, and solar radiation & temperature. Then, sample from the PDF to create scenarios for specified number and use K-means to reduce the number of scenarios for 8760 hours.

 - `scenario_generation_duration.py`: Scenario generation with duration curve of demand.

