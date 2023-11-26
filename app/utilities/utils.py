import os
import time
import shutil
import torch
import pandas as pd

from app import logger
from pathlib import Path


def check_gpu():
    if torch.cuda.is_available():
        logger.info(f"Cuda available! Running on GPU!")
        return True
    
    if torch.backends.mps.is_available():
        logger.info(f"MPS available! Running on GPU!")
        return True

    logger.critical(f"GPU is not available. Running on CPU!")

    return False


def copy_uploaded_file(full_file, filename):
    time_nanosec = time.time_ns()
    temp_path = os.path.join(
        os.getenv("TEST_PATH"),
        f"{str(time_nanosec)}",
        f"{filename}"
    )
    Path(os.path.dirname(temp_path)).mkdir(
        parents=True, 
        exist_ok=True
    )
    logger.info(f"Copying {filename} to {temp_path}")

    with open(f'{temp_path}',"wb") as buffer:
        shutil.copyfileobj(full_file, buffer)

    return temp_path


def preprocess_text(text):
    new_text = []

    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        # t = '' if t.startswith(('@', 'http')) and len(t) > 1 else t
        new_text.append(t)
    return " ".join(new_text)


def create_output_csv(output_path, output_data):
    df = pd.DataFrame(output_data)
    csv_output_path = os.path.join(
        output_path,
        "output_sentiment_test.csv"
    )
    return df.to_csv(
        csv_output_path, 
        index=False
    )


def create_results_txt_file(
    output_path,
    model_name,
    preprocessing,
    labels,
    df_shape,
    metrics
):
    file_path = os.path.join(
        output_path,
        "results.txt"
    )
    with open(file_path, "w") as file:
        file.write(f"Model ID: {model_name}\n")
        file.write(f"Text Preprocessing: {preprocessing}\n")
        file.write(f"Expected Labels: {labels}\n")
        file.write(f"Dataframe Length: {df_shape}\n\n")
        file.write(f"-----------------------------------")
        file.write(f"\n\nAccuracy: {metrics['accuracy']}\n\n")
        file.write("Classification Report:\n")
        file.write(metrics["classification report"])
        file.write(f"\n\nConfusion Matrix: \n\n{metrics['confusion matrix']}")
    return