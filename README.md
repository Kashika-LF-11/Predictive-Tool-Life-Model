# Predictive-Tool-Life-Model
A complete repository for training, preprocessing, and deploying the Predictive Tool Life Model. Includes scripts for training and inference, real-time &amp; batch preprocessing, and detailed documentation for the infra team.


## Repository Structure
Predictive-Tool-Life-Model/
├── README.md
├── docs/
│   └── Predictive Tool Life Model.pdf
├── sagemaker_training/
│   ├── src/ 
│       └── train.py
│   └── build_and_push.sh
│   └── create_training_job.py
│   └── Dockerfile
│   └── job_config.yaml
│   └── Pipfile
│   └── Pipfile.lock
├── data_preprocessing/
│   ├── preprocess_inference.py
│   └── preprocess_train.py
└── inference_function.py 


---


### `docs/`  
Contains documentation related to the Tool Life Model.  
- `Tool_Life_Model_Documentation.pdf`: Detailed description of the model architecture, workflows, and usage.

### `inference_function.py`  
The **inference function** used to call the trained model and return predictions.  
It outputs a tuple of `(prediction, reliability_score)`.
Used because the basic .predict() method will not inherently return a reliability_score. 
This function describes the (prediction + reliability_score) implementation. 

### `sagemaker_training/`  
Self-contained directory with **AWS SageMaker training flow scripts**.  
This directory is designed to be used directly, without depending on any other folder or file in the repository.  
It contains all scripts required for training the Tool Life Model on SageMaker.

### `data_preprocessing/`  
Contains preprocessing scripts for both training and inference data:
- `preprocess_inference.py`: Preprocessing flow for **real-time, single-data-point inference**.
- `preprocess_train.py`: Preprocessing flow for **batch data** to prepare training datasets.

---

