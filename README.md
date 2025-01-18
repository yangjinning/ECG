# ECG

## STEP 1

To install all dependencies:
``` 
conda create -n inbatchcl python=3.10
conda activate inbatchcl
conda install mpi4py
pip install -r requirements.txt
```


## STEP 2

Unzip data in dataset\mimic-iv-ecg\data


## STEP 3

Download Llama-3.1-8B from HuggingFace and save it in $local_llama_filepath

https://huggingface.co/meta-llama/Llama-3.1-8B


## STEP 4

Begin training:
``` 
python run_main.py --local_llm_path $local_llama_filepath
``` 


