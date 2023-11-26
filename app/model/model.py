import os
import numpy as np

from scipy.special import softmax
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification
)

from app import logger, labels
from app.utilities.utils import (
    check_gpu,
    preprocess_text
)

class Backbone:
    def __init__(
        self, 
        model_name
    ):
        model_root_path = os.getenv("MODEL_PATH")
        self.model_id = model_name
        self.model_path = f"{model_root_path}/{model_name}"


    async def setup(self):
        logger.info(f"Model Initializing...")
        self.gpu = check_gpu()
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()
    

    def get_tokenizer(self):
        logger.info("Initializing Tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id
        )
        logger.info("Tokenizer successfully initialized!")
        return tokenizer


    def get_config(self):
        logger.info("Initializing Config ...")
        config = AutoConfig.from_pretrained(
            self.model_id
        )
        logger.info("Config successfully initialized!")
        return config


    def get_model(self):
        logger.info(f"Getting Model {self.model_id}  from {os.path.dirname(self.model_path)}...")

        from_local = False
        model_src = self.model_id

        if os.path.exists(self.model_path):
            from_local = True
            model_src = self.model_path

        model = AutoModelForSequenceClassification.from_pretrained(
            model_src,
            local_files_only=from_local
        )

        logger.info("Model successfully initialized!\n\n")
        return model


    async def infer(
        self,
        text,
        preprocess=True
    ):  
        logger.info(f"Predicting sentiment of tweet: '{text}' ...")

        if preprocess:
            text = preprocess_text(text)
            logger.info(f"Processed Tweet: '{text}'")

        encoded_input = self.tokenizer(
            text, 
            return_tensors='pt'
        )

        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        top_score = round(float(max(scores))*100, 2)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        output = ranking[0]
        label = labels[output]
        
        logger.critical(
            f"** Predicted sentiment: '{label}' ||  Confidence score: '{top_score}' **"
        )

        return {
            "Input": text,
            "Output": {
                "model_output": label,
                "confidence_score": top_score
            }
        }