import os 
import asyncio

from app import logger
from app.model.model import Backbone
from app.model.eval import BatchEval

model_handler = None

async def initialize_model(instance):
    logger.info("Booting up ModelHandler...")

    global model_handler
    model_handler = Backbone(
        instance.model_name
    )
    task = asyncio.create_task(
        model_handler.setup()
    )

    done, pending = await asyncio.wait({task})
    
    if (
        task in done and 
        "model" in dir(model_handler)
    ):  
        logger.info("---- Sentiment Analyzer is ready! ----\n\n")
        os.environ["MODEL"]="true"
    

async def infer(text, preprocess):  
    if model_handler is None:
        return -2

    answer = await asyncio.gather(
        model_handler.infer(
            text,
            preprocess
        )
    )
    try:
        return answer
    except Exception as e:
        pass


async def batch_eval(
    file,
    preprocess
):
    if model_handler is None:
        return -2
    
    eval_handler = BatchEval(
        file=file, 
        model_id=model_handler.model_id,
        model=model_handler.model,
        tokenizer=model_handler.tokenizer,
        preprocess=preprocess
    )
    task = asyncio.create_task(
        eval_handler.evaluate()
    )

    done, pending = await asyncio.wait({task})
    ouput_path = os.listdir(eval_handler.output_path)

    if (
        task in done and
        "output.csv" in ouput_path and
        "output.txt" in ouput_path
    ):
        logger.info(f"Evaluation results saved under --> {eval_handler.output_path}")