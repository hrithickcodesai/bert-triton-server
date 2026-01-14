import os
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import BertTokenizer


class TritonPythonModel:
    def initialize(self, args):
        path = os.path.join(args["model_repository"], args["model_version"])
        self.tokenizer = BertTokenizer.from_pretrained(path)

    def execute(self, requests):
        responses = []
        for request in requests:
            query = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
            query_decoded = [q.decode("utf-8") for q in query.flatten()]

            encoded = self.tokenizer(
                query_decoded,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )

            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)
            token_type_ids = encoded["token_type_ids"].astype(np.int64)

            out_input_ids = pb_utils.Tensor("INPUT_IDS", input_ids)
            out_attention_mask = pb_utils.Tensor("ATTENTION_MASK", attention_mask)
            out_token_type_ids = pb_utils.Tensor("TOKEN_TYPE_IDS", token_type_ids)

            responses.append(
                pb_utils.InferenceResponse(
                    [out_input_ids, out_attention_mask, out_token_type_ids]
                )
            )

        return responses
