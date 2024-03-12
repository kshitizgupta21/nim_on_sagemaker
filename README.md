## Steps for running on sagemaker

### 1. Build/Pull docker container
```
# authenticate
docker login nvcr.io
# username is $oauthtoken
# password is <NGC API Key>
```
Then build docker
```
cd final_docker
bash build_docker.sh nim-24.02-sm-final
```

### 2. Tag and push to ECR
```
cd ../
bash push_ecr.sh nim-24.02-sm-final
```

### 3. Download the prebuilt optimized model from NGC
Install ngc-cli
```
wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.39.0/zip -O ngc_cli_3.39.0.zip
unzip ngc_cli_3.39.0.zip
unzip ngccli_linux.zip
chmod u+x ngc-cli
export PATH=$PATH:ngc-cli
```

Set up ngc config
```
ngc config set
```

This is expected output. Might have to select `nvidian/nemo-llm` as the org and team
```
(base) [ec2-user@ip-172-16-120-240 ~]$ ngc config set
CLI_VERSION: Latest - 3.40.0 available (current: 3.39.0). Please update by using the command 'ngc version upgrade' 
Enter API key [no-apikey]. Choices: [<VALID_APIKEY>, 'no-apikey']: <VALID_APIKEY>
Enter CLI output format type [ascii]. Choices: ['ascii', 'csv', 'json']: 
Enter org [no-org]: nvidian
Enter team [no-team] : nemo-llm
Enter ace [no-ace] : [no-ace] 
Validating configuration...
Successfully validated configuration.
Saving configuration...
Successfully saved NGC configuration to /home/ec2-user/.ngc/config
```

To see the list of all available prebuilt models:
```
ngc registry model list "nvidian/nemo-llm/*"
```


This is the expected format for the `model download-version`
```
ngc registry model download-version "{Repository}:{Latest Version}"
```

Here we download LLama-2-7B engine for 1-A100 GPU
```
ngc registry model download-version "nvidian/nemo-llm/llama-2-7b-chat:LLAMA-2-7B-CHAT-4K-FP16-1-A100.24.02.rc2
```

### 4. Run through the notebook to create SM endpoint
Run `nim_sm_prebuilt_a100.ipynb` notebook as SageMaker notebook instance on `g5.xlarge` instance. For the sagemaker notebook instance you can use any instance type.
