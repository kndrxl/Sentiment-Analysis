import os
import pandas as pd
import numpy as np

from tqdm import tqdm 
from scipy.special import softmax
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score
)

from app import logger, labels
from app.utilities.utils import (
    copy_uploaded_file,
    create_output_csv,
    preprocess_text,
    create_results_txt_file
)

class BatchEval:
    def __init__(
        self,
        file,
        model_id,
        model,
        tokenizer,
        preprocess=True
    ):  
        self.model=model
        self.model_id=model_id
        self.tokenizer=tokenizer
        self.preprocess=preprocess
        self.csv_path = copy_uploaded_file(
            file.file,
            file.filename
        )
        self.output_path=os.path.dirname(self.csv_path)

    def get_metrics(
        self,
        predictions, 
        true_labels
    ):
        cls_report = classification_report(
            true_labels, 
            predictions
        )
        cnf_matrix = confusion_matrix(
            true_labels, 
            predictions
        )
        accuracy = accuracy_score(true_labels, predictions)

        return {
            "accuracy": accuracy,
            "classification report": cls_report,
            "confusion matrix": cnf_matrix
        }


    def infer(
        self,
        text
    ):  
        if self.preprocess:
            text = preprocess_text(text)

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

        return {
            "tweet": text,
            "sentiment": label,
            "score": top_score
        }


    async def evaluate(self):  
        logger.info(f"Evaluating ...")

        df = pd.read_csv(self.csv_path)
        data = df['text'].tolist()
        true_labels = df['expected_sentiment'].tolist()

        output_data={
            "text": data,
            "expected_sentiment": true_labels,
            "model_output": [],
            "confidence_score": []
        }

        for i in tqdm(range(len(data)), desc="Processing"):
            text = data[i]
            result = self.infer(text)
            predicted_label = result['sentiment']
            score = result['score']
            output_data["model_output"].append(predicted_label)
            output_data["confidence_score"].append(score)

        metrics = self.get_metrics(
            output_data["model_output"], 
            true_labels
        )

        create_output_csv(
            self.output_path, 
            output_data
        )
        create_results_txt_file(
            self.output_path,
            self.model_id,
            self.preprocess,
            labels,
            len(df),
            metrics
        )

        logger.info(f"\n\n--- Evaluation done! Results saved under --> {self.output_path} ---")