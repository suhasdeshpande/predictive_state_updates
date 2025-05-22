#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv(override=True)

from pydantic import BaseModel
from crewai.flow import start
from crewai import LLM
import logging
from copilotkit.crewai import (
    CopilotKitFlow,
    tool_calls_log,
    FlowInputState
)
from typing import List, Dict, Any
import json

WRITE_DOCUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "write_document",
        "description": " ".join("""
            Write a document. Use markdown formatting to format the document.
            It's good to format the document extensively so it's easy to read.
            You can use all kinds of markdown.
            However, do not use italic or strike-through formatting, it's reserved for another purpose.
            You MUST write the full document, even when changing only a few words.
            When making edits to the document, try to make them minimal - do not change every word.
            Keep stories SHORT!
            """.split()),
        "parameters": {
            "type": "object",
            "properties": {
                "document": {
                    "type": "string",
                    "description": "The document to write"
                },
            },
        }
    }
}


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

class PredictiveStateFlowInputState(FlowInputState):
    """Defines the expected input state for the PredictiveStateUpdateFlow."""
    document: str = ""


class PredictiveStateUpdateFlow(CopilotKitFlow[PredictiveStateFlowInputState]):

    @start()
    def chat(self):
        """
        Standard chat node.
        """
        system_prompt = f"""
        You are a helpful assistant for writing documents.
        To write the document, you MUST use the write_document tool.
        You MUST write the full document, even when changing only a few words.
        When you wrote the document, DO NOT repeat it as a message.
        Just briefly summarize the changes you made. 2 sentences max.
        This is the current state of the document: ----\n {self.state.document}\n-----
        """

        logger.info(f"System prompt: {system_prompt}")

        # Initialize CrewAI LLM with streaming enabled
        # CrewAI's LLM class expects 'model' as the parameter name
        llm = LLM(model="gpt-4o", stream=True)

        # Get message history using the base class method
        # This should now correctly use self.state.messages from AgentInputState
        messages = self.get_message_history(system_prompt=system_prompt)

        try:
            # Track tool calls
            initial_tool_calls_count = len(tool_calls_log)
            logger.info(f"Initial tool calls count: {initial_tool_calls_count}")
            response_content = llm.call(
                messages=messages,
                tools=[WRITE_DOCUMENT_TOOL],
                available_functions={"write_document": self.write_document_handler}
            )

            logger.info(f"Response content: {response_content}")

            # Handle tool responses using the base class method
            final_response = self.handle_tool_responses(
                llm=llm,
                response_text=response_content, # Pass the text content of the response
                messages=messages, # Original messages sent to LLM
                tools_called_count_before_llm_call=initial_tool_calls_count
            )

            # ---- Maintain conversation history ----
            # 1. Add the current user message(s) to conversation history
            for msg in self.state.messages:
                if msg.get('role') == 'user' and msg not in self.state.conversation_history:
                    self.state.conversation_history.append(msg)

            # 2. Add the assistant's response to conversation history
            assistant_message = {"role": "assistant", "content": final_response}
            self.state.conversation_history.append(assistant_message)


            return json.dumps({
                "response": final_response,
                "id": self.state.id
            })

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            return f"\n\nAn error occurred: {str(e)}\n\n"

    def write_document_handler(self, document):
        # Update the document in state
        logger.info(f"#### Handling write_document tool call ####: {document}")
        self.state.document = document
        return document


def kickoff():
    predictive_state_update_flow = PredictiveStateUpdateFlow()
    predictive_state_update_flow.kickoff({
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": "Write a document about the history of the internet."
                }
            ],
            "document": ""  # Initialize document field
        }
    })

def plot():
    predictive_state_update_flow = PredictiveStateUpdateFlow()
    predictive_state_update_flow.plot()


if __name__ == "__main__":
    kickoff()
