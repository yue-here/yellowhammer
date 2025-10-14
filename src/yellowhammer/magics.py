"""
Magics to support LLM interactions in IPython/Jupyter.
Inspired by fperez/jupytee and jan-janssen/LangSim.
"""

import os
import re
import base64
import pprint

# Configure Logfire if token is available
try:
    import logfire
    logfire_token = os.getenv("LOGFIRE_TOKEN")
    if logfire_token:
        logfire.configure(token=logfire_token)
except ImportError:
    pass  # Logfire is optional

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
from IPython.display import Markdown, HTML
from datalab_api import DatalabClient
from .fulltext_search import FullTextSearchTool
from .llm import datalab_agent, Deps
from .prompt import API_PROMPT, SYSTEM_PROMPT
from .utils import process_prompt_with_images
import asyncio
import nest_asyncio
import pprint
import rich
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.syntax import Syntax
from rich.panel import Panel
from rich.box import ROUNDED
from io import StringIO

nest_asyncio.apply()

@magics_class
class DatalabMagics(Magics):
    def __init__(self, shell):
        super().__init__(shell)

        self.messages = None
        self.item_manifest = None
        self.search_tool = None

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
    @argument(
        "-T",
        "--temp",
        type=float,
        default=0.1,
        help="""Temperature, float in [0,1]. Higher values push the algorithm
        to generate more aggressive/"creative" output. [default=0.1].""",
    )
    @line_cell_magic
    def llm(self, line, cell=None):
        """
        Multimodal LLM interaction with Jupyter magics.
        Supports text prompts and local image files.
        """
        args = parse_argstring(self.llm, line)

        # Get the prompt
        if cell is None:
            prompt = " ".join(args.prompt)
        else:
            prompt = cell

        # Process prompt for images and file paths
        messages, warnings = process_prompt_with_images(prompt)
        
        # Display any warnings to the user
        for warning in warnings:
            print(f"⚠️  {warning}")

        # Retrieve item manifest using get_items() and initialize search tool if needed
        if self.item_manifest is None or self.search_tool is None:
            with DatalabClient(os.getenv("DATALAB_URL")) as client:
                self.item_manifest = client.get_items(item_type="samples")
                self.search_tool = FullTextSearchTool(self.item_manifest)

        item_manifest = self.item_manifest
        search_tool = self.search_tool
        
        # Run the datalab agent with processed messages (including images)
        response = datalab_agent.run_sync(
            messages,  # May include both text and BinaryContent for images
            deps=Deps(
                client=DatalabClient,
                datalab_api_key=os.getenv("DATALAB_API_KEY"),
                datalab_url=os.getenv("DATALAB_URL"),
                item_manifest=item_manifest,
                search_tool=search_tool,
                ),
            message_history=self.messages,
            model_settings={'temperature': args.temp},
        )

        self.messages = response.all_messages()

        if hasattr(response.data, 'text'):
            # Create a string IO to capture Rich console output
            console_output = StringIO()
            console = Console(file=console_output, width=120, highlight=True)
            
            # Format main response text with Rich Markdown
            console.print(RichMarkdown(response.data.text))
            
            # Add usage info in a panel with syntax highlighting
            usage_str = pprint.pformat(response.usage())
            usage_syntax = Syntax(usage_str, "python", theme="monokai", word_wrap=True)
            console.print(Panel(usage_syntax, title="Usage Information", border_style="blue", box=ROUNDED))
            
        # Output code into next cell
        if hasattr(response.data, 'code'):
            cell_fill = response.data.code
            get_ipython().set_next_input(cell_fill)
