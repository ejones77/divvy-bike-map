# divvy-bike-map
Cloud service for predicting the availability of Divvy bikes in Chicago


## Dependencies
- uv (python 3.12)
- Go (1.25.0)
- AWS
    - S3
    - This project also uses EC2, which you can set up to emulate production
- Neon (or other serverless postgres provider)
  - Or for a quick-start, download data from kaggle
- Docker
- Make (for local dev) 

## Setup on your own
- clone the repo
- set up env vars
  - copy this file in the ml-pipeline directory
- Set up UV
    - cd ml-pipeline && uv sync
- download from kaggle, placing json files in `ml-pipeline/data`
    - If you want to keep collecting data, you upsert the data to Neon or your postgres provider of choice. 
- train the model
    - while still in ml-pipeline
    - ```uv run --env-file .env python divvy_ml/pipelines/xgb_trainer.py``` 
- This should output a folder `ml-pipeline/xgb_model_DATE` -- save this model to an S3 bucket
- in the main directory of the repo, ```make up```

- the map should appear in localhost! 

## Tests



## Monitoring

