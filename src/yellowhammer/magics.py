"""
Magics to support LLM interactions in IPython/Jupyter.
Inspired by fperez/jupytee and jan-janssen/LangSim.
"""

import os
import re
import base64
import pprint

from IPython import get_ipython
from IPython.core.magic import (
    Magics,
    magics_class,
    line_cell_magic,
)
from IPython.core.magic_arguments import (
    magic_arguments,
    argument,
    parse_argstring,
)
from IPython.display import Markdown
# from langchain_core.prompts import ChatPromptTemplate
from datalab_api import DatalabClient
from .llm_pydantic import datalab_agent, Deps
from .prompt import API_PROMPT, SYSTEM_PROMPT
import asyncio
import nest_asyncio
nest_asyncio.apply()

def parse_paths(input_string):
    # A simple regex pattern to match file-path-like substrings:
    # [A-Za-z0-9_.\\/-]+\.[A-Za-z0-9_]+
    #    - one or more letters, digits, underscores, dots, slashes, or dashes,
    #      containing at least one dot that leads to a plausible extension
    file_path_pattern = r"([A-Za-z0-9_.\\/-]+\.[A-Za-z0-9_]+)"

    # Find all file paths
    paths = re.findall(file_path_pattern, input_string)

    # Remove those file paths from the original string
    remaining_text = re.sub(file_path_pattern, "", input_string)

    # Clean up extra whitespace
    remaining_text = re.sub(r"\s+", " ", remaining_text).strip()

    return {"text": remaining_text, "paths": paths}

# Class to manage state and expose the main magics
@magics_class
class DatalabMagics(Magics):
    def __init__(self, shell):
        super().__init__(shell)

        self.messages = None
        # self.images = []  # Initialize with empty list of images.

    # A datalab magic that returns a code block
    @magic_arguments()
    @argument(
        "prompt",
        nargs="*",
        help="""LLM prompt. When used as a line magic,
        it runs to the end of the line. In cell mode, the entire cell
        is considered the code generation prompt.
        """,
    )
    # @argument(
    #     "-T",
    #     "--temp",
    #     type=float,
    #     default=0.1,
    #     help="""Temperature, float in [0,1]. Higher values push the algorithm
    #     to generate more aggressive/"creative" output. [default=0.1].""",
    # )
    @line_cell_magic
    def llm(self, line, cell=None):
        """
        Multimodal LLM interaction with Jupyter magics
        """
        args = parse_argstring(self.llm, line)  # self.llm is a bound method

        # Parse the prompt to extract any image paths
        if cell is None:
            prompt = " ".join(args.prompt)
        else:
            prompt = cell
        prompt = parse_paths(prompt)
        prompt_text = prompt["text"]
        paths = prompt["paths"]  # TODO implement check or make parser only output local paths

        # Run the datalab agent
        response = datalab_agent.run_sync(
            prompt_text,
            deps=Deps(
            DatalabClient,
            os.getenv("DATALAB_API_KEY"),
            os.getenv("DATALAB_URL"),
            ),
            message_history=self.messages,
        )

        self.messages = response.all_messages()

        # Output text and code
        if hasattr(response.data, 'code'):
            cell_fill = response.data.code
            get_ipython().set_next_input(cell_fill)

        if hasattr(response.data, 'text'):
            usage = pprint.pformat(response.usage())
            return Markdown(response.data.text + "\n" + usage)
        
        return "Unable to generate response."

        #             # generate variable name "image_n" and inject it into a message
        #             image_variable_name = f"image_{len(self.images)}"
        #             image_message = [
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {
        #                         "url": f"data:image/jpeg;base64,{{{image_variable_name}}}"
        #                     },
        #                 }
        #             ]
        #             self.messages.append(("human", image_message))
        #         except FileNotFoundError:
        #             print(f"Error: Could not read image file {path[1]}")
        #             pass

        #         # Add the image message to the message list

        # # Create a dict from self.images to pass the image data to langchain
        # image_dict = {f"image_{i+1}": image for i, image in enumerate(self.images)}
