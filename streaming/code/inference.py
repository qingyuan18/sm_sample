# -*- coding: utf-8 -*-
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import json
import uuid
import io
import sys

import traceback
from PIL import Image
import requests
import boto3
import sagemaker
import torch
from fastapi import FastAPI, Request
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from torch import autocast
#from transformers import T5Tokenizer, T5ForConditionalGeneration
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

class ChatGLM():

    def clear(self) -> None:
        if torch.cuda.is_available():
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

    def stream(self, query, history):
        if query is None or history is None:
            yield {"query": "", "response": "", "history": [], "finished": True}
        size = 0
        response = ""
        for response, history in self.model.stream_chat(self.tokenizer, query, history):
            this_response = response[size:]
            history = [list(h) for h in history]
            size = len(response)
            yield {"delta": this_response, "response": response, "finished": False}
        logger.info("Answer - {}".format(response))
        yield {"query": query, "delta": "[EOS]", "response": response, "history": history, "finished": True}


    def getLogger(name, file_name, use_formatter=True):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s    %(message)s')
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        if file_name:
            handler = logging.FileHandler(file_name, encoding='utf8')
            handler.setLevel(logging.INFO)
            if use_formatter:
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
                handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


    def __init__(self) -> None:
        logger.info("Start initialize model...")
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        self.model = self._model(quantize_level, gpu_id)
        self.model.eval()
        _, _ = self.model.chat(self.tokenizer, "你好", history=[])
        logger.info("Model initialization finished.")

    def _model(self):
        return model_fn(None)

    def clear(self) -> None:
        if torch.cuda.is_available():
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

    def answer(self, query: str, history):
        response, history = self.model.chat(self.tokenizer, query, history=history)
        history = [list(h) for h in history]
        return response, history



    def model_fn(model_dir):
        print("=================model_fn_Start=================")
        model_s3_path = os.environ['MODEL_S3_PATH']
        print("=================model s3 path=================="+model_s3_path)
        os.system("cp ./code/s5cmd  /tmp/ && chmod +x /tmp/s5cmd")
        os.system("/tmp/s5cmd sync {0} {1}".format(model_s3_path+"*","/tmp/model/"))
        if os.environ["MODEL_TYPE"] == "ptuning":
            config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, pre_seq_len=128)
            model = AutoModel.from_pretrained("THUDM/chatglm-6b", config=config, trust_remote_code=True)
            prefix_state_dict = torch.load(os.path.join("/tmp/model/", "pytorch_model.bin"))
            new_prefix_state_dict = {}
            for k, v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

        elif os.environ["MODEL_TYPE"] == "full turning":
            print("====================load full turning=================")
            config = AutoConfig.from_pretrained("/tmp/model/", trust_remote_code=True, pre_seq_len=128)
            model = AutoModel.from_pretrained("/tmp/model/", trust_remote_code=True)
        else:
            print("====================load normal ======================")
            config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
            model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

        #model = model.to("cuda")
        model = model.quantize(4)
        model = model.half().cuda()
        model = model.eval()
        print("=================model_fn_End=================")
        return model


    def output_fn(prediction, content_type):
        """
        Serialize and prepare the prediction output
        """
        print(content_type)
        return json.dumps(
            {
                'answer': prediction
            }
        )



