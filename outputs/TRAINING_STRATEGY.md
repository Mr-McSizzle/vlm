# Training Strategy (UPDATED POST-ACQUISITION)

## Dataset Selection and Strategy
Given the 4GB VRAM constraint and the tight 23-hour hackathon timeline, we strictly prioritize minimum viable data to unblock the pipeline without spending hours waiting for large archives to download over streaming. 

1. **VQA (Priority 1): ~20 examples**
   - Supplied by `dmarsili/RSVQA-LR-2k` (validation split).
   - *Why:* Core interaction mode for the VLM.

2. **Change-VQA (Priority 2): ~40 examples**
   - Supplied by `ljx620/CDVQA`.
   - *Why:* Temporal comparison is a high-value remote sensing capability that base LLaVA completely lacks.

3. **Captioning / Domain Adaptation: ~20 examples**
   - Supplied by `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`.
   - *Why:* BigEarthNet serves as "domain adaptation" to help the model learn satellite terminology. Since `BigEarthNet.txt` provides only text annotations, we pass them as text-only domain-adaptation records to the language model directly without images.

4. **Grounding: OMITTED (Currently 0 examples)**
   - *Why:* Grounding annotations exist in `xiang709/VRSBench`. However, the raw image archive `Images_val.zip` alone is 3.9 GB. Downloading this completely blocked our CI environment limits and bandwidth. We have disabled VRSBench in `vlm/config.yaml` to successfully Unblock the rest of the pipeline using the data that is feasible to retrieve.

## Constraints
- **Bandwidth/Time:** We downloaded aggressive subsets (20-40 examples per task) via HuggingFace APIs to bypass massive streaming latencies.
- **Current State:** The datasets (RSVQA, CDVQA, BigEarthNet) are physically downloaded locally to `data/external/`. The unified JSON pipeline was generated, schema-validated, leak-checked, and is fully functional.
- **Status:** DATA PIPELINE UNBLOCKED.
