import os

from typing import Literal, Optional, List
from fastapi import UploadFile, HTTPException
from pydantic import (
    BaseModel,
    validator,
    root_validator
)

class Instance(BaseModel):
    model_name: Literal[
        'citizenlab/twitter-xlm-roberta-base-sentiment-finetunned',
        'cardiffnlp/twitter-roberta-base-sentiment',
        'cardiffnlp/twitter-xlm-roberta-base-sentiment',
        'cardiffnlp/bertweet-base-sentiment',
        'finiteautomata/bertweet-base-sentiment-analysis',
        'cardiffnlp/twitter-roberta-base-sentiment-latest'
    ]='cardiffnlp/twitter-roberta-base-sentiment-latest'


class Message(BaseModel):
    text: str
    preprocess: bool=True


class BatchEval(BaseModel):
    file: UploadFile
    preprocess: bool=True

    @validator('file')
    def check_file_type(cls, file):
        if file is None:
            raise HTTPException(
                status_code=400, 
                detail=f"No File Uploaded!"
            )
        filename = str(file.filename)
        if not filename.lower().endswith((".csv")):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsuported filetype - '{filename}'"
            )
        return file