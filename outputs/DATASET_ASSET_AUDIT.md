# Dataset Asset Audit

This report documents the physical presence of image assets referenced by the `data/unified` manifest records.

## Audit Results
- **Total Records across all splits:** 240
- **Total Image References:** 180
- **Existing Image References:** 0
- **Missing Image References:** 180

## Missing Records by Dataset
- **RSVQA:** 100
- **CDVQA:** 40
*(Note: BigEarthNet.txt records have 0 image references, so they do not fail the image existence check)*

## Missing Records by Split
- **train:** 114
- **val:** 10
- **test:** 16

## Conclusion and Cause
The local mock environment does not contain the `data/external` directory. All 180 image files required by the RSVQA and CDVQA tasks are physically missing. 
Since acquiring the massive datasets entirely is impractical under hackathon constraints, and `download_subsets.py` limits are set to 20 per dataset by default, the most robust fix is to re-download a minimal subset of images using `download_subsets.py` and then strictly prune any unified dataset records whose images remain missing.
