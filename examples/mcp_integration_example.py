#!/usr/bin/env python3
"""
Example script demonstrating MCP (Model Context Protocol) integration with LangChain.

This script shows how to use the MCPTool and MCPResourceTool to interact with
MCP servers from within LangChain applications.
"""

import json
import logging
from typing import Any, Dict

from langchain_core.tools import BaseTool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockMCPTool(BaseTool):
    """Mock MCP tool for demonstration purposes."""
    
    name: str = "mcp_tool"
    description: str = """Execute tools from an MCP server. Available tools:
- file_read: Read a file from the filesystem
- file_write: Write content to a file
- calculator: Perform mathematical calculations

Provide input as JSON with 'tool_name' and 'arguments' fields."""
    
    def _run(self, query: str, **kwargs) -> str:
        """Mock tool execution."""
        try:
            parsed = json.loads(query)
            tool_name = parsed["tool_name"]
            arguments = parsed.get("arguments", {})
            
            if tool_name == "file_read":
                filename = arguments.get("filename", "unknown")
                return f"Contents of {filename}: This is mock file content."
            
            elif tool_name == "file_write":
                filename = arguments.get("filename", "unknown")
                content = arguments.get("content", "")
                return f"Successfully wrote {len(content)} characters to {filename}"
            
            elif tool_name == "calculator":
                expression = arguments.get("expression", "0")
                # Simple eval for demo - don't use in production!
                try:
                    result = eval(expression)
                    return f"Result: {result}"
                except Exception as e:
                    return f"Error evaluating expression: {e}"
            
            else:
                return f"Error: Unknown tool '{tool_name}'. Available tools: file_read, file_write, calculator"
                
        except json.JSONDecodeError:
            return "Error: Invalid JSON input"
        except Exception as e:
            return f"Error: {e}"
    
    async def _arun(self, query: str, **kwargs) -> str:
        """Async version - delegates to sync for this example."""
        return self._run(query, **kwargs)


class MockMCPResourceTool(BaseTool):
    """Mock MCP resource tool for demonstration purposes."""
    
    name: str = "mcp_resource"
    description: str = """Access resources from an MCP server. Available resources:
- file://documents/readme.txt (readme.txt): Project documentation
- file://data/config.json (config.json): Configuration file
- file://logs/app.log (app.log): Application logs

Provide the resource URI as input."""
    
    def _run(self, query: str, **kwargs) -> str:
        """Mock resource access."""
        uri = query.strip()
        
        if uri == "file://documents/readme.txt":
            return """# Project Documentation
This is a sample project that demonstrates MCP integration with LangChain.

## Features
- File operations
- Mathematical calculations
- Resource access

## Usage
Use the MCP tools to interact with the system."""
        
        elif uri == "file://data/config.json":
            return json.dumps({
                "server_url": "http://localhost:8000",
                "timeout": 30,
                "debug": True
            }, indent=2)
        
        elif uri == "file://logs/app.log":
            return """2024-01-15 10:00:00 INFO Starting MCP server
2024-01-15 10:00:01 INFO Server listening on port 8000
2024-01-15 10:00:02 INFO MCP tools initialized
2024-01-15 10:00:03 INFO Ready to accept connections"""
        
        else:
            available = ["file://documents/readme.txt", "file://data/config.json", "file://logs/app.log"]
            return f"Error: Resource '{uri}' not found. Available resources: {', '.join(available)}"
    
    async def _arun(self, query: str, **kwargs) -> str:
        """Async version - delegates to sync for this example."""
        return self._run(query, **kwargs)


def demonstrate_mcp_tools():
    """Demonstrate basic MCP tool usage."""
    print("🔧 MCP Tool Integration Example")
    print("=" * 40)
    
    # Create mock tools
    mcp_tool = MockMCPTool()
    mcp_resource = MockMCPResourceTool()
    
    print("\n1. Using MCP Tool for calculations:")
    calc_input = json.dumps({
        "tool_name": "calculator", 
        "arguments": {"expression": "25 * 4 + 10"}
    })
    print(f"Input: {calc_input}")
    result = mcp_tool._run(calc_input)
    print(f"Output: {result}")
    
    print("\n2. Using MCP Tool for file operations:")
    file_input = json.dumps({
        "tool_name": "file_write",
        "arguments": {
            "filename": "example.txt",
            "content": "Hello from MCP integration!"
        }
    })
    print(f"Input: {file_input}")
    result = mcp_tool._run(file_input)
    print(f"Output: {result}")
    
    print("\n3. Using MCP Resource Tool:")
    resource_uri = "file://documents/readme.txt"
    print(f"Input: {resource_uri}")
    result = mcp_resource._run(resource_uri)
    print(f"Output:\n{result}")
    
    print("\n4. Error handling example:")
    invalid_input = '{"invalid": "json structure"}'
    print(f"Input: {invalid_input}")
    result = mcp_tool._run(invalid_input)
    print(f"Output: {result}")


def demonstrate_real_mcp_usage():
    """Demonstrate how to use real MCP tools."""
    print("\n🌐 Real MCP Integration Example")
    print("=" * 40)
    
    print("""
To use real MCP tools with LangChain:

1. Start an MCP server (e.g., filesystem server, database server, etc.)
2. Initialize the MCPTool with the server URL:

from langchain_community.tools.mcp.tool import MCPTool, MCPResourceTool

# Connect to a real MCP server
mcp_tool = MCPTool(server_url="http://localhost:8000")

# Use in your LangChain application
tools = [mcp_tool]

3. The tool will automatically discover available MCP tools and resources
4. Agents can then use these tools through the standard LangChain interface

Example MCP servers you can try:
- File system operations
- Database queries  
- Web scraping
- API integrations
- And more!
""")


def main():
    """Main function to run all examples."""
    print("🦜 LangChain MCP Integration Examples")
    print("=" * 50)
    
    # Demonstrate basic tool usage
    demonstrate_mcp_tools()
    
    # Show real usage instructions
    demonstrate_real_mcp_usage()
    
    print("\n✅ Examples completed!")
    print("\nTo learn more about MCP (Model Context Protocol):")
    print("- Visit: https://spec.modelcontextprotocol.io/")
    print("- GitHub: https://github.com/modelcontextprotocol")


if __name__ == "__main__":
    main()