# Notebook Example for deploying NVIDIA NIM on sagemaker

### 0. Launch SageMaker notebook instance

Launch SageMaker notebook instance on `ml.t3.medium` instance with **role `AmazonSageMakerServiceCatalogProductsUseRole`** and then git clone this repo inside that instance.

### 1. Pull and Build SageMaker compatible NIM docker container
In terminal of SageMaker notebook instance run the following commands. First we authenticate into NGC
```
# authenticate into NGC
docker login nvcr.io
# username is $oauthtoken
# password is <NGC API Key>
```
Then pull docker for specific LLM NIM
* LLama3-8B = nvcr.io/nim/meta/llama3-8b-instruct
* LLama3-70B = nvcr.io/nim/meta/llama3-70b-instruct

```
#export SRC_IMAGE_PATH=nvcr.io/nim/meta/llama3-70b-instruct
export SRC_IMAGE_PATH=nvcr.io/nim/meta/llama3-8b-instruct

export SRC_IMAGE_NAME="${SRC_IMAGE_PATH##*/}"
export SRC_IMAGE_NAME="${SRC_IMAGE_NAME%%:*}"
docker pull ${SRC_IMAGE_PATH}
docker tag ${SRC_IMAGE_PATH} ${SRC_IMAGE_NAME}

awk -v src="$SRC_IMAGE_PATH" '{gsub(/{{ SRC_IMAGE }}/, src)}1' Dockerfile > Dockerfile.nim
docker build -f Dockerfile.nim -t nim-${SRC_IMAGE_NAME} .
```

### 2. Tag and push SM compatible NIM container to ECR
```
bash push_ecr.sh nim-${SRC_IMAGE_NAME}
```

### 3. Run `nim_llama3_8b_a10g.ipynb` notebook to deploy NIM LLama-3 8B on g5.xlarge (A10G GPU) on SageMaker

### 4 Run `nim_llama3_70b_a100.ipynb` notebook to deploy LLama-3 70B model on p4d.24xlarge (A100 GPU) on SageMaker



