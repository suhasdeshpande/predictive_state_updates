#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from typing import Any
from crewai import LLM
from crewai.flow import start
from pydantic import BaseModel, Field
from copilotkit.crewai import (
    CopilotKitFlow,
    tool_calls_log,
    FlowInputState,
    emit_copilotkit_state_update_event
)
from crewai.flow import persist

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


class Document(BaseModel):
    """A document with title and content."""
    title: str = Field(..., description="The title of the document")
    content: str = Field(..., description="The markdown content of the document")

class AgentState(FlowInputState):
    """The state of the document."""
    document: Document = None

@persist()
class DocumentWritingFlow(CopilotKitFlow[AgentState]):

    @start()
    def chat(self):
        """Standard chat node."""
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

        # Initialize CrewAI LLM with streaming enabled
        llm = LLM(model="gpt-4o", stream=True)

        # Get message history using the base class method
        messages = self.get_message_history(system_prompt=system_prompt)
        # Get available tools using the base class method
        # This should now correctly use self.state.tools from AgentInputState
        tools_definitions = self.get_available_tools()

        # Format tools for OpenAI API using the base class method
        formatted_tools, available_functions = self.format_tools_for_llm(tools_definitions)

        try:
            # Track tool calls
            initial_tool_calls_count = len(tool_calls_log)

            response_content = llm.call(
                messages=messages,
                tools=[
                    WRITE_DOCUMENT_TOOL,
                    *formatted_tools
                ],
                available_functions={
                    "write_document": self.write_document_handler,
                    **available_functions
                }
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
        self.state.document = document_obj

        # Fire state update event
        try:
            emit_copilotkit_state_update_event(
                tool_name="write_document",
                args={"document": self.state}
            )
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
