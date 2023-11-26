import os
import asyncio

from app import logger
from fastapi import (
    FastAPI, 
    BackgroundTasks, 
    Depends
)
from app.src.handler import (
    initialize_model, 
    infer,
    batch_eval
)
from app.utilities.validator import(
    Instance,
    Message,
    BatchEval
)

os.environ["MODEL"]=""

app = FastAPI(
    title="Twitter Sentiment Analysis",
    root_path=os.environ.get("OPENAPI_PREFIX", "")
)

@app.get("/")
def healthcheck():
    return {"Message": "Health - OK!"}


@app.post("/initialize")
async def model_initialize(
    background_tasks: BackgroundTasks,
    instance: Instance=Depends()
):
    try:
        background_tasks.add_task(
            initialize_model, 
            instance
        )
        return {
            "Message" : f"Initializing with backbone '{instance.model_name}'!"
        }
    
    except Exception as e:
        logger.error("\033[91m{}\033[0m".format(
            "MAIN | Exception encountered: {}".format(repr(e))
        ))
        return {"Error" : "Failed to initialized"}


@app.post("/infer")
async def single_inference(message: Message=Depends()):
    try:
        if (
            os.getenv("MODEL")==""
        ):
            return {"Error" : "Model not yet initialized!"}
        
        res = await asyncio.gather(infer(
            message.text,
            message.preprocess
        ))
        return res[0][0]

    except Exception as e:

        logger.error("\033[91m{}\033[0m".format(
            "MAIN | Exception encountered: {}".format(repr(e))
        ))

        return {"Error" : "Failed to interpret the message"}
    

@app.post("/batch_eval")
async def batch_evaluation(
    background_tasks: BackgroundTasks,
    file: BatchEval=Depends()
):
    try:
        if os.getenv("MODEL")=="":
            return {"Error" : "Model not yet initialized!"}
        
        background_tasks.add_task(
            batch_eval,
            file.file,
            file.preprocess
        )
        return {'message': 'Evaluation process on-going in the background. Monitor the logs.'}
    
    except Exception as e:
        logger.error("\033[91m{}\033[0m".format(
            "MAIN | Exception encountered: {}".format(repr(e))
        ))
        return {"Error" : "Failed to evaluate"}