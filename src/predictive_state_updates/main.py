#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv(override=True)

from pydantic import BaseModel
from crewai.flow import Flow, start
from crewai import LLM
from pprint import pprint
import logging
import json

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

class PredictiveStateUpdateState(BaseModel):
    sentence_count: int = 1
    poem: str = ""


class PredictiveStateUpdateFlow(Flow[PredictiveStateUpdateState]):

    @start()
    def chat(self):
        return "Hello, world!"


def kickoff():
    predictive_state_update_flow = PredictiveStateUpdateFlow()
    predictive_state_update_flow.kickoff()

def plot():
    predictive_state_update_flow = PredictiveStateUpdateFlow()
    predictive_state_update_flow.plot()


if __name__ == "__main__":
    kickoff()
