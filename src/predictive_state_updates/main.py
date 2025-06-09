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
    """Event for state updates in CopilotKit"""
    type: str = "copilotkit_state_update"
    tool_name: str
    args: dict[str, Any]
    timestamp: str = Field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())

class Document(BaseModel):
    """A document with title and content."""
    title: str = Field(..., description="The title of the document")
    content: str = Field(..., description="The markdown content of the document")

class AgentState(FlowInputState):
    """The state of the document."""
    data: dict[str, Any] = Field(default_factory=lambda: {"document": None}, description="Data containing document")

@persist()
class DocumentWritingFlow(CopilotKitFlow[AgentState]):

    @start()
    def chat(self):
        """Standard chat node."""
        current_doc_info = "No document created yet"
        if self.state.data.get("document"):
            current_doc_info = f"Document: {self.state.data['document']}"

        system_prompt = f"""
        You are a helpful assistant for writing documents.
        To write or modify a document, you MUST use the write_document tool.
        You MUST write the full document, even when changing only a few words.
        When you wrote the document, DO NOT repeat it as a message.
        Just briefly summarize the changes you made. 2 sentences max.
        This is the current state of the document: ----\n {current_doc_info}\n-----
        """

        # Initialize CrewAI LLM with streaming enabled
        llm = LLM(model="gpt-4o", stream=True)

        # Get message history using the base class method
        messages = self.get_message_history(system_prompt=system_prompt)

        try:
            # Track tool calls
            initial_tool_calls_count = len(tool_calls_log)

            response_content = llm.call(
                messages=messages,
                tools=[WRITE_DOCUMENT_TOOL],
                available_functions={"write_document": self.write_document_handler}
            )

            # Handle tool responses using the base class method
            final_response = self.handle_tool_responses(
                llm=llm,
                response_text=response_content,
                messages=messages,
                tools_called_count_before_llm_call=initial_tool_calls_count
            )

            # Maintain conversation history
            for msg in self.state.messages:
                if msg.get('role') == 'user' and msg not in self.state.conversation_history:
                    self.state.conversation_history.append(msg)

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
        # Store the full document as string in data.document
        self.state.data["document"] = f"Title: {document_obj.title}\n\nContent:\n{document_obj.content}"

        # Fire state update event
        try:
            state_update_event = CopilotKitStateUpdateEvent(
                tool_name="write_document",
                args={"document": self.state.data["document"]}
            )
            crewai_event_bus.emit(None, event=state_update_event)
        except Exception as e:
            logger.error(f"Error emitting state update event: {e}")

        return "Document written successfully."

def kickoff():
    """Initialize and run the document writing flow"""
    document_flow = DocumentWritingFlow()

    inputs = {
        "inputs": {
            "state": {
                "id": "",
                "timestamp": 0,
                "source": "",
                "data": {"document": None}
            },
            "messages": []
        }
    }

    result = document_flow.kickoff(inputs)
    print(result)

def plot():
    """Generate flow visualization"""
    document_flow = DocumentWritingFlow()
    document_flow.plot()

if __name__ == "__main__":
    kickoff()
