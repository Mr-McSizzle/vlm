# Final Quantitative Evaluation

## Overall Accuracy
- **Base:** 62.5%
- **Adapted:** 25.0%
- **Improvement:** -37.5%

## Task Accuracy
- **Base VQA:** 50.0% | **Adapted VQA:** 0.0%
- **Base Change-VQA:** 75.0% | **Adapted Change-VQA:** 50.0%

## Deltas
- **Improved:** 1
- **Degraded:** 4
- **Unchanged Correct:** 1
- **Unchanged Incorrect:** 2

## Qualitative Failure Categories
- **Counting Failure**: Adapted model outputs 0 or 2 for large groups of objects, failing to count correctly (e.g. expected 104, answered 0).
- **Yes/No Classification Failure**: Model defaults to 'yes' indiscriminately for Change-VQA and VQA, regardless of actual change.
- **Temporal/Change Reasoning Failure**: The model fails to compare spatial structures over time, instead adopting a lazy 'yes' bias.
- **Hallucination**: Hallucinating objects (like oceans or squares) that are not physically present in the test image.
- **Truncation**: Output cutting off mid-sentence (mitigated by configuring `max_new_tokens` correctly, but present in earlier runs).
- **Unsupported Explanation**: The model adds irrelevant reasoning to simple answers, reducing precision.

## Per-Example Results
| ID | Task | Expected | Base Answer | Adapted Answer | Base Correct | Adapted Correct |
|---|---|---|---|---|---|---|
| cdvqa_69 | change_vqa | yes | I cannot tell. no. | yes | False | True |
| rsvqa_13 | vqa | 10 | There are 12 water areas visible. | 2 | False | False |
| rsvqa_17 | vqa | no | no | yes | True | False |
| cdvqa_28 | change_vqa | yes | yes | no | True | False |
| cdvqa_31 | change_vqa | yes | yes | yes, regions changed | True | True |
| cdvqa_35 | change_vqa | no | The images are identical. no change. | yes, the playgrounds changed | True | False |
| rsvqa_3 | vqa | 104 | 15 buildings can be seen. | 0 | False | False |
| rsvqa_14 | vqa | no | no, there is no circular building. | yes | True | False |