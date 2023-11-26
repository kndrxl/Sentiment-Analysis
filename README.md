## Prerequisites
- [Conda](https://docs.conda.io/en/latest/)
- Python >=3.10.12
- MacOS Apple Silicon Chip (M1/M2/M3) or Ubuntu 16.04+


## Setup

Clone the repository to working directory.

```bash
git clone git@github.com:kndrxl/Sentiment-Analysis.git

cd Sentiment-Analysis
```

Download models --> [here](https://drive.google.com/file/d/14XXS5jTmEhtXGQzyar5x55nJpnQ-KYfZ/view?usp=sharing) and extract the zip file to the root directory of the repo.


```bash
unzip models.zip -d models/
```

The final folder structure should be like this:

```bash
.
├── app
│   ├── __init__.py
│   ├── model
│   │   ├── eval.py
│   │   └── model.py
│   ├── src
│   │   ├── handler.py
│   │   └── main.py
│   └── utilities
│       ├── utils.py
│       └── validator.py
├── data
│   ├── AI Technical Task.docx
│   └── sentiment_test_cases.csv
├── models
│   └── cardiffnlp
│       ├── twitter-roberta-base-sentiment
│       │   ├── config.json
│       │   └── pytorch_model.bin
│       ├── twitter-roberta-base-sentiment-latest
│       │   ├── config.json
│       │   └── pytorch_model.bin
│       └── twitter-xlm-roberta-base-sentiment
│           ├── config.json
│           └── pytorch_model.bin
├── notebooks
│   ├── twitter-roberta-base-sentiment-latest.ipynb
│   ├── twitter-roberta-base-sentiment.ipynb
│   └── twitter-xlm-roberta-base-sentiment.ipynb
├── output
│   └── <DATETIME>
│       ├── output_sentiment_test.csv
│       ├── results.txt
│       └── sentiment_test_cases.csv
├── README.md
├── requirements.txt
├── run.sh
├── .gitignore
├── .env

```


## Local Installation (MacOS)
- Create a conda environment.

    ```bash
    conda create --name <ENV_NAME> python=3.10.12
    ```

- Activate the environment.

    ```bash
    conda activate <ENV_NAME>
    ```

- Install packages and dependencies.

    ```bash
    pip install -r requirements.txt --no-cache-dir
    ```

## Initialization

Run via FastAPI:

Either run the shell script

```bash
./run.sh
```

or

```bash
uvicorn app.src.main:app --port 8080     
```

Once app is running, open the OpenAPI Swagger UI here --> [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs)

Choose and initialize a model first via this endpoint: [/model_initialize](http://127.0.0.1:8080/docs#/default/model_initialize_initialize_post). So far, the best model is the default `cardiffnlp/twitter-roberta-base-sentiment-latest`.

The initialization is an asynchronous process so monitor the logs via terminal.

## Testing

- Once Model is ready, we can now test [/infer](http://127.0.0.1:8080/docs#/default/single_inference_infer_post) (`single inference`) and [/batch_eval](http://127.0.0.1:8080/docs#/default/batch_evaluation_batch_eval_post) (`batch evaluation`) endpoints.

- For single inference, the required parameters are:
    - text: str - Input Tweet/Text
    - preprocess: bool - Option for preprocessing input.

- For batch evaluation, the required parameters are:
    - file: UploadFile - CSV file to be uploaded via user-interface.
    - preprocess: bool - Option for preprocessing input texts.
    
- The results of batch evaluation will be saved under the [output](output/) folder.