"""MCP (Model Context Protocol) tools for LangChain.

This module provides integration with MCP servers, enabling LangChain to use
tools and resources exposed via the Model Context Protocol.
"""

from langchain_community.tools.mcp.tool import MCPTool, MCPResourceTool

__all__ = ["MCPTool", "MCPResourceTool"]