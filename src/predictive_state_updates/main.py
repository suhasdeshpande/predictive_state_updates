#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from typing import Optional, Any
from crewai import LLM
from crewai.flow import start
from pydantic import BaseModel, Field
from copilotkit.crewai import (
    CopilotKitFlow,
    tool_calls_log,
    FlowInputState
)
from crewai.utilities.events.base_events import BaseEvent
from crewai.flow import persist
from crewai.utilities.events import crewai_event_bus

WRITE_DOCUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "write_document",
        "description": (
            "Write or modify a document. Use markdown formatting extensively. "
            "You MUST write the full document, even when changing only a few words. "
            "When making edits, try to make them minimal - do not change every word. "
            "Keep content concise and well-structured. "
            "Do not use italic or strike-through formatting."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "document": {
                    "description": "The document object containing all details.",
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title of the document.",
                        },
                        "content": {
                            "type": "string",
                            "description": "The full markdown content of the document.",
                        },
                    },
                    "required": ["title", "content"],
                }
            },
            "required": ["document"],
        },
    },
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CopilotKitStateUpdateEvent(BaseEvent):
    """
    Event for state updates in CopilotKit
    """
    type: str = "copilotkit_state_update"
    tool_name: str
    args: dict[str, Any]
    timestamp: str = Field(default_factory=lambda: __import__('datetime').datetime.now().isoformat(), description="Event timestamp")

    def __init__(self, **data):
        super().__init__(**data)

class Document(BaseModel):
    """
    A document with title and content.
    """
    title: str = Field(..., description="The title of the document")
    content: str = Field(..., description="The markdown content of the document")

class AgentState(FlowInputState):
    """
    The state of the document.
    """
    document: Optional[str] = None

@persist()
class DocumentWritingFlow(CopilotKitFlow[AgentState]):

    def __init__(self, *args, **kwargs):
        print(f"=== INIT CALLED === args: {args}")
        print(f"=== INIT CALLED === kwargs: {kwargs}")
        logger.info(f"DocumentWritingFlow.__init__ called with args: {args}")
        logger.info(f"DocumentWritingFlow.__init__ called with kwargs: {kwargs}")
        super().__init__(*args, **kwargs)
        print(f"=== INIT COMPLETED === State: {getattr(self, 'state', 'No state')}")
        logger.info(f"DocumentWritingFlow.__init__ completed. State: {getattr(self, 'state', 'No state')}")

    @start()
    def chat(self):
        """
        Standard chat node.
        """
        print(f"=== CHAT START === State: {self.state}")
        logger.info(f"=== CHAT START === State: {self.state}")

        current_doc_info = "No document created yet"
        if self.state.document:
            current_doc_info = f"Document: {self.state.document}"

        system_prompt = f"""
        You are a helpful assistant for writing documents.
        To write or modify a document, you MUST use the write_document tool.
        You MUST write the full document, even when changing only a few words.
        When you wrote the document, DO NOT repeat it as a message.
        Just briefly summarize the changes you made. 2 sentences max.
        This is the current state of the document: ----\n {current_doc_info}\n-----
        """

        logger.info(f"System prompt: {system_prompt}")

        # Initialize CrewAI LLM with streaming enabled
        llm = LLM(model="gpt-4o", stream=True)

        # Get message history using the base class method
        messages = self.get_message_history(system_prompt=system_prompt)

        # For testing: Add the user message directly since kickoff doesn't pass it correctly
        test_user_message = {"role": "user", "content": "Write a document about Dogs."}
        if not any(m.get('role') == 'user' for m in messages):
            messages.append(test_user_message)


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
                response_text=response_content,
                messages=messages,
                tools_called_count_before_llm_call=initial_tool_calls_count
            )

            # Check if tools were actually called
            final_tool_calls_count = len(tool_calls_log)
            tools_called = final_tool_calls_count - initial_tool_calls_count
            logger.info(f"Tools called during this interaction: {tools_called}")

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
        """Handler for the write_document tool"""
        # Convert the document dict to a Document object for validation
        document_obj = Document(**document)
        # Store the full document as string (matching AgentState type)
        self.state.document = f"Title: {document_obj.title}\n\nContent:\n{document_obj.content}"
        print(f"=== DOCUMENT STORED === \n {self.state.document}")

        # Fire state update event
        try:
            print("=== ABOUT TO CREATE STATE UPDATE EVENT ===")
            state_update_event = CopilotKitStateUpdateEvent(
                tool_name="write_document",
                args={"document": self.state.document}
            )
            print(f"=== STATE UPDATE EVENT CREATED: {state_update_event} ===")
            logger.info(f"Emitting state update event with document: {document_obj.title}")
            print("=== ABOUT TO EMIT EVENT ===")
            crewai_event_bus.emit(None, event=state_update_event)
            print("=== EVENT EMITTED ===")
        except Exception as e:
            print(f"=== ERROR IN STATE UPDATE: {e} ===")
            logger.error(f"Error emitting state update event: {e}")

        return document_obj.model_dump_json(indent=2)

def kickoff():
    logger.info("=== KICKOFF STARTING ===")
    document_flow = DocumentWritingFlow()
    logger.info("=== DocumentWritingFlow CREATED ===")

    inputs = {
        "inputs": {
            "state": { "document": 'Life of dog' },
            "messages": [
                {
                    "id": "ck-687c4250-1dd9-4a56-bd49-c96f068c7a78",
                    "role": "user",
                    "content": "Write a document."
                }
            ]
        }
    }
    logger.info(f"=== INPUTS: {inputs} ===")

    result = document_flow.kickoff(inputs)
    logger.info(f"=== RESULT: {result} ===")
    print(f"FINAL RESULT: {result}")  # Use print to ensure it shows up


def plot():
    document_flow = DocumentWritingFlow()
    document_flow.plot()

if __name__ == "__main__":
    kickoff()
