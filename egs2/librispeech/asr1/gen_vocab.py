from datasets import load_dataset
import re

chars_to_remove_regex = "[\,\?\.\!\-\;\:\"\“\%\‘\”\�']"


def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, "", batch["sentence"]).lower()
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


common_voice_train = load_dataset("common_voice", "fr", split="train+validation")
common_voice_test = load_dataset("common_voice", "fr", split="test")

common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_test = common_voice_test.map(remove_special_characters)

vocab_train = common_voice_train.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=common_voice_train.column_names,
)
vocab_test = common_voice_test.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=common_voice_test.column_names,
)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
print(vocab_dict)
