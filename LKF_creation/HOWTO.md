<div align="center">

#### CUSTOM PARAPHRASING

- Model used for paraphrasing: Mistral-7B-Instruct-v0.2
- Run fix_tofu for all sets: this would create paraphrased json files.
- Use these json's to creat HF-datasets using pert_muse/arrow.py
- Copy the produced arrow to MUSE/TOFU local HF-location.
- Update the feature keys in dataset_info.json on the same local HF-location.

</div>