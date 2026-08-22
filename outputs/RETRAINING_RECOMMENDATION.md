# Retraining Recommendation

**A. Is the current adapted model better than the base model?**
No. The adapted model is demonstrably worse. The overall accuracy dropped from 62.5% to 25.0%. The model lost foundational reasoning and counting capabilities, defaulting instead to a naive "yes" bias for binary questions and failing entirely at numeric estimation (e.g., answering 0 when the expected count was 104).

**B. Is the difference statistically/meaningfully supported by this tiny test set?**
No, it is not statistically significant. The test set is absurdly small (only 8 records). A single changed answer shifts the accuracy by 12.5%. However, the *qualitative* degradation (loss of counting ability, lazy biases) across these 8 examples is a meaningful indicator of severe overfitting and catastrophic forgetting caused by training on an equally tiny 64-example dataset.

**C. Which task is weakest?**
VQA is the weakest task. The adapted model scored 0.0% on VQA (down from 50.0% base accuracy). It failed all counting questions (producing answers like "0" or "2" for counts in the tens or hundreds) and flipped correct "no" classifications into incorrect "yes" classifications. Change-VQA also degraded, dropping from 75.0% to 50.0% due to the same indiscriminate "yes" bias.

**D. Is more training data required?**
Yes, absolutely. A training corpus of 64 total examples is completely inadequate for fine-tuning an 8B parameter vision-language model. The current dataset size induces severe overfitting and mode collapse (such as always outputting "yes").

**E. What should the next training dataset contain?**
The next training dataset must be significantly scaled up to prevent mode collapse and teach generalizable reasoning. It should contain:
- **500+ diverse Change-VQA pairs** with a balanced distribution of "yes" (change occurred) and "no" (no change) examples to break the "yes" bias.
- **500+ standard VQA examples**, heavily featuring counting tasks with varying magnitudes (small, medium, and large numbers) and spatial relationship tasks.
- **Valid Image-Caption pairs** (since BigEarthNet was corrupted and removed), ensuring every caption corresponds to a single real image and contains descriptive natural language, not bounding boxes.

**F. Is another T4 training run worth doing?**
**No**, not until the dataset is radically expanded. Running another T4 sequence on the same 64-example dataset will just yield the exact same overfitted, collapsed model. Time and compute resources should be spent on data acquisition (e.g., retrieving actual VRSBench and larger CDVQA subsets) before scheduling another GPU session.
