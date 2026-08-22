# Retraining Recommendation

## Corrected Dataset Size
- **Total Records:** 80 (64 train, 8 val, 8 test)
- **VQA:** 20
- **Change-VQA:** 60
- **Caption:** 0
- **Grounding:** 0

## Data Removals
- **BigEarthNet (Caption)**: 20 records were entirely removed from the unified dataset. The text descriptions were merely bounding box coordinates (e.g., `[0.0 0.33, 0.28 0.8]`), and the records contained absolutely zero physical image references (`"images": []`). This fundamentally violates image-caption training.
- **Grounding**: 0 records exist because real VRSBench image assets are unavailable.

## Recommendation for Another Training Run
**The current corrected training corpus of 64 training examples is TOO SMALL for robust, generalized domain adaptation.**

While the pipeline is now completely free of evaluation bugs, static image duplication, truncated generations, and corrupt training data, fine-tuning an 8B/7B parameter model on 64 examples will primarily cause aggressive overfitting (memorization) rather than teaching the model the core semantic tasks of remote sensing Change-VQA. 

### Is another T4 run worth the available time?
**No**, unless the dataset is scaled first. 
Running a T4 training sequence on 64 examples will successfully prove that the *mechanics* of the LoRA pipeline work, but it will not yield a model capable of zero-shot generalizability on unseeen CDVQA test records.

**Minimum Practical Training Configuration**:
We need a minimum of 500+ Change-VQA pairs and 500+ VQA images to expect any meaningful alignment in the adapter weights. If a run *must* be done to prove the hackathon pipeline operates end-to-end, the 64-record set can be used, but expectations for qualitative performance improvements should be strictly bounded.
