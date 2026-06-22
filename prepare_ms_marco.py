"""
準備 MS MARCO v1.1 子集並上傳到 Hugging Face Hub。

用法:
    1. pip install datasets huggingface_hub
    2. huggingface-cli login
    3. python prepare_ms_marco.py
"""

from datasets import load_dataset, DatasetDict

# ============ 設定 ============
SOURCE_DATASET = "microsoft/ms_marco"
SOURCE_VERSION = "v1.1"
SUBSET_SIZE = 10_000
HF_REPO = "Lala8383/ms-marco-qa-10k" 
SEED = 42
# ==============================


def main():
    print(f"正在載入 {SOURCE_DATASET} ({SOURCE_VERSION})...")
    ds = load_dataset(SOURCE_DATASET, SOURCE_VERSION)

    # 從 train split 隨機抽樣
    train = ds["train"].shuffle(seed=SEED).select(range(min(SUBSET_SIZE, len(ds["train"]))))

    # validation 保留完整（本身就不大）或也切一部分
    val_size = min(1000, len(ds["validation"]))
    validation = ds["validation"].shuffle(seed=SEED).select(range(val_size))

    # test split
    test_size = min(1000, len(ds["test"]))
    test = ds["test"].shuffle(seed=SEED).select(range(test_size))

    subset = DatasetDict({
        "train": train,
        "validation": validation,
        "test": test,
    })

    print(f"子集大小: train={len(subset['train'])}, validation={len(subset['validation'])}, test={len(subset['test'])}")
    print(f"欄位: {subset['train'].column_names}")
    print(f"範例:\n{subset['train'][0]}")

    # 上傳到 Hugging Face Hub
    print(f"\n正在上傳到 {HF_REPO}...")
    subset.push_to_hub(
        HF_REPO,
        private=False,
        commit_message="Upload MS MARCO v1.1 QA subset (10k)",
    )
    print("上傳完成！")

    # 建立 dataset card
    create_dataset_card()


def create_dataset_card():
    card = """\
---
language:
  - en
license: other
license_name: microsoft-research-license
license_link: https://microsoft.github.io/msmarco/
tags:
  - question-answering
  - ms-marco
  - reading-comprehension
source_datasets:
  - microsoft/ms_marco
task_categories:
  - question-answering
dataset_info:
  config_name: default
pretty_name: MS MARCO QA Subset (10K)
---

# MS MARCO QA Subset (10K)

This is a **subset** of the [MS MARCO v1.1 dataset](https://huggingface.co/datasets/microsoft/ms_marco) by Microsoft, sampled for lightweight experimentation.

## Source

- **Original dataset**: [microsoft/ms_marco](https://huggingface.co/datasets/microsoft/ms_marco) (v1.1)
- **Original paper**: [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/abs/1611.09268)
- **Original authors**: Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, Li Deng (Microsoft)

## What was changed

- Randomly sampled **10,000** examples from the train split (seed=42)
- Randomly sampled **1,000** examples from the validation split (seed=42)
- No other modifications were made to the data

## License

This dataset is derived from MS MARCO, which is released under the
[Microsoft Research License](https://microsoft.github.io/msmarco/).
Please refer to the original license terms before use.

## Citation

If you use this dataset, please cite the original MS MARCO paper:

```bibtex
@article{nguyen2016ms,
  title={MS MARCO: A Human Generated MAchine Reading COmprehension Dataset},
  author={Nguyen, Tri and Rosenberg, Mir and Song, Xia and Gao, Jianfeng and Tiwary, Saurabh and Majumder, Rangan and Deng, Li},
  journal={arXiv preprint arXiv:1611.09268},
  year={2016}
}
```
"""
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    print("Dataset card 已上傳！")


if __name__ == "__main__":
    main()
