import os
import torch
import onnx
from transformers import BertModel, BertTokenizer


class BertCLS(torch.nn.Module):
    def __init__(self, model_name: str):
        super(BertCLS, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        output = outputs.last_hidden_state[:, 0, :]
        return output


def export():
    model_name = "bert-base-uncased"
    output_dir = "model_repository"

    tokenizer_dir = os.path.join(output_dir, "bert_tokenizer/1")
    model_dir = os.path.join(output_dir, "bert_model/1")
    ensemble_dir = os.path.join(output_dir, "bert_ensemble/1")

    os.makedirs(tokenizer_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(ensemble_dir, exist_ok=True)

    print(f"Exporting tokenizer to {tokenizer_dir}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tokenizer_dir)

    print(f"Exporting model to {model_dir}...")
    model = BertCLS(model_name)
    model.eval()

    dummy_input = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "token_type_ids": torch.zeros(2, 8, dtype=torch.long),
    }

    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]
    token_type_ids = dummy_input["token_type_ids"]

    onnx_path = os.path.join(model_dir, "model.onnx")

    # dynamic axes for variable
    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["cls_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "cls_output": {0: "batch_size", 1: "hidden_size"},
        },
        opset_version=17,
    )

    # change the IR version to align with Triton compatibility
    print("Downgrading ONNX IR version to 9 for Triton compatibility...")
    onnx_model = onnx.load(onnx_path)
    onnx_model.ir_version = 9
    onnx.save(onnx_model, onnx_path)
    print("Export complete.")

    m = onnx.load("model_repository/bert_model/1/model.onnx")
    for o in m.graph.output:
        print(
            o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]
        )


if __name__ == "__main__":
    export()
