FROM nvcr.io/nvidia/tritonserver:23.10-py3
RUN pip install --no-cache-dir transformers
COPY model_repository /models
CMD ["tritonserver", "--model-repository=/models"]