from fastapi import FastAPI
from time import sleep
import uvicorn
from datetime import datetime
from fastapi import Request


from inference import *


app = FastAPI()
app.add_middleware( CORSMiddleware,
        allow_origins = ["*"],
        allow_credentials = True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

bot = ChatGLM()
MAX_HISTORY = 5

@app.get('/ping')
async def ping():
    return {"message": "ok"}


@app.post("/stream")
def answer_question_stream(arg_dict: dict):
    def decorate(generator):
        for item in generator:
            yield ServerSentEvent(json.dumps(item, ensure_ascii=False), event='delta')
    result = {"query": "", "response": "", "success": False}
    try:
        text = arg_dict["query"]
        ori_history = arg_dict["history"]
        logger.info("Query - {}".format(text))
        if len(ori_history) > 0:
            logger.info("History - {}".format(ori_history))
        history = ori_history[-MAX_HISTORY:]
        history = [tuple(h) for h in history]
        return EventSourceResponse(decorate(bot.stream(text, history)))
    except Exception as e:
        logger.error(f"error: {e}")
        return EventSourceResponse(decorate(bot.stream(None, None)))



@app.post('/invocations')
def answer_question_stream(arg_dict: dict):
    def decorate(generator):
        for item in generator:
            yield ServerSentEvent(json.dumps(item, ensure_ascii=False), event='delta')
    result = {"query": "", "response": "", "success": False}
    try:
        text = arg_dict["query"]
        ori_history = arg_dict["history"]
        logger.info("Query - {}".format(text))
        if len(ori_history) > 0:
            logger.info("History - {}".format(ori_history))
        history = ori_history[-MAX_HISTORY:]
        history = [tuple(h) for h in history]
        return EventSourceResponse(decorate(bot.stream(text, history)))
    except Exception as e:
        logger.error(f"error: {e}")
        return EventSourceResponse(decorate(bot.stream(None, None)))




