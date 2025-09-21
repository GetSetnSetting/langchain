"""MCP Tool implementation for LangChain."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, ValidationError

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from .client import MCPClient

logger = logging.getLogger(__name__)


class MCPToolSchema(BaseModel):
    """Schema for MCP tool input validation."""
    pass


class MCPTool(BaseTool):
    """Tool for interacting with MCP (Model Context Protocol) servers.
    
    This tool enables LangChain to call tools exposed by MCP servers,
    providing a bridge between LangChain agents and MCP-compatible services.
    """
    
    name: str = "mcp_tool"
    description: str = (
        "Execute tools from an MCP (Model Context Protocol) server. "
        "Provide the tool name and arguments as JSON."
    )
    
    mcp_client: MCPClient = Field(exclude=True)
    available_tools: List[Dict[str, Any]] = Field(default_factory=list)
    
    def __init__(self, server_url: str, **kwargs):
        """Initialize the MCP tool.
        
        Args:
            server_url: URL of the MCP server
            **kwargs: Additional keyword arguments
        """
        mcp_client = MCPClient(server_url)
        available_tools = []
        
        try:
            # Fetch available tools from the MCP server
            tools = mcp_client.list_tools()
            available_tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema
                }
                for tool in tools
            ]
            
            # Update description with available tools
            if available_tools:
                tool_descriptions = []
                for tool in available_tools:
                    tool_descriptions.append(f"- {tool['name']}: {tool['description']}")
                
                description = (
                    f"Execute tools from an MCP server. Available tools:\n"
                    + "\n".join(tool_descriptions) +
                    "\n\nProvide input as JSON with 'tool_name' and 'arguments' fields."
                )
            else:
                description = (
                    "Execute tools from an MCP server. "
                    "No tools currently available. "
                    "Provide input as JSON with 'tool_name' and 'arguments' fields."
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize MCP tool: {e}")
            description = (
                "Execute tools from an MCP server. "
                "Error connecting to server. "
                "Provide input as JSON with 'tool_name' and 'arguments' fields."
            )
        
        super().__init__(
            mcp_client=mcp_client,
            available_tools=available_tools,
            description=description,
            **kwargs
        )
    
    def _parse_input(self, tool_input: str) -> Dict[str, Any]:
        """Parse the tool input string.
        
        Args:
            tool_input: JSON string containing tool_name and arguments
            
        Returns:
            Parsed input dictionary
        """
        try:
            parsed = json.loads(tool_input)
            if not isinstance(parsed, dict):
                raise ValueError("Input must be a JSON object")
            
            if "tool_name" not in parsed:
                raise ValueError("Input must contain 'tool_name' field")
            
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input: {e}")
    
    def _validate_tool_exists(self, tool_name: str) -> bool:
        """Validate that the requested tool exists on the MCP server.
        
        Args:
            tool_name: Name of the tool to validate
            
        Returns:
            True if tool exists, False otherwise
        """
        return any(tool["name"] == tool_name for tool in self.available_tools)
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the MCP tool.
        
        Args:
            query: JSON string containing tool_name and arguments
            run_manager: Callback manager for the tool run
            
        Returns:
            Result from the MCP tool execution
        """
        try:
            # Parse input
            parsed_input = self._parse_input(query)
            tool_name = parsed_input["tool_name"]
            arguments = parsed_input.get("arguments", {})
            
            # Validate tool exists
            if not self._validate_tool_exists(tool_name):
                available_tool_names = [tool["name"] for tool in self.available_tools]
                return (
                    f"Error: Tool '{tool_name}' not found. "
                    f"Available tools: {', '.join(available_tool_names)}"
                )
            
            # Call the MCP tool
            result = self.mcp_client.call_tool(tool_name, arguments)
            
            # Format the result
            if isinstance(result, dict):
                return json.dumps(result, indent=2)
            else:
                return str(result)
                
        except ValueError as e:
            return f"Input validation error: {str(e)}"
        except Exception as e:
            logger.error(f"MCP tool execution failed: {e}")
            return f"Tool execution failed: {str(e)}"
    
    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Async version of _run. Currently delegates to sync version."""
        return self._run(query, run_manager)
    
    def refresh_tools(self) -> None:
        """Refresh the list of available tools from the MCP server."""
        try:
            tools = self.mcp_client.list_tools(force_refresh=True)
            self.available_tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema
                }
                for tool in tools
            ]
            
            # Update description
            if self.available_tools:
                tool_descriptions = []
                for tool in self.available_tools:
                    tool_descriptions.append(f"- {tool['name']}: {tool['description']}")
                
                self.description = (
                    f"Execute tools from an MCP server. Available tools:\n"
                    + "\n".join(tool_descriptions) +
                    "\n\nProvide input as JSON with 'tool_name' and 'arguments' fields."
                )
            else:
                self.description = (
                    "Execute tools from an MCP server. "
                    "No tools currently available. "
                    "Provide input as JSON with 'tool_name' and 'arguments' fields."
                )
                
        except Exception as e:
            logger.error(f"Failed to refresh MCP tools: {e}")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools.
        
        Returns:
            List of available tools with their metadata
        """
        return self.available_tools.copy()


class MCPResourceTool(BaseTool):
    """Tool for accessing MCP resources."""
    
    name: str = "mcp_resource"
    description: str = (
        "Access resources from an MCP (Model Context Protocol) server. "
        "Provide the resource URI as input."
    )
    
    mcp_client: MCPClient = Field(exclude=True)
    available_resources: List[Dict[str, Any]] = Field(default_factory=list)
    
    def __init__(self, server_url: str, **kwargs):
        """Initialize the MCP resource tool.
        
        Args:
            server_url: URL of the MCP server
            **kwargs: Additional keyword arguments
        """
        mcp_client = MCPClient(server_url)
        available_resources = []
        
        try:
            # Fetch available resources from the MCP server
            resources = mcp_client.list_resources()
            available_resources = [
                {
                    "uri": resource.uri,
                    "name": resource.name,
                    "mime_type": resource.mime_type,
                    "description": resource.description
                }
                for resource in resources
            ]
            
            # Update description with available resources
            if available_resources:
                resource_descriptions = []
                for resource in available_resources:
                    desc = f"- {resource['uri']} ({resource['name']})"
                    if resource['description']:
                        desc += f": {resource['description']}"
                    resource_descriptions.append(desc)
                
                description = (
                    f"Access resources from an MCP server. Available resources:\n"
                    + "\n".join(resource_descriptions) +
                    "\n\nProvide the resource URI as input."
                )
            else:
                description = (
                    "Access resources from an MCP server. "
                    "No resources currently available. "
                    "Provide the resource URI as input."
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize MCP resource tool: {e}")
            description = (
                "Access resources from an MCP server. "
                "Error connecting to server. "
                "Provide the resource URI as input."
            )
        
        super().__init__(
            mcp_client=mcp_client,
            available_resources=available_resources,
            description=description,
            **kwargs
        )
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Access an MCP resource.
        
        Args:
            query: URI of the resource to access
            run_manager: Callback manager for the tool run
            
        Returns:
            Content of the resource
        """
        try:
            uri = query.strip()
            
            # Validate resource exists
            if not any(resource["uri"] == uri for resource in self.available_resources):
                available_uris = [resource["uri"] for resource in self.available_resources]
                return (
                    f"Error: Resource '{uri}' not found. "
                    f"Available resources: {', '.join(available_uris)}"
                )
            
            # Read the resource
            result = self.mcp_client.read_resource(uri)
            
            # Format the result
            if isinstance(result, dict):
                return json.dumps(result, indent=2)
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"MCP resource access failed: {e}")
            return f"Resource access failed: {str(e)}"
    
    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Async version of _run. Currently delegates to sync version."""
        return self._run(query, run_manager)
    
    def get_available_resources(self) -> List[Dict[str, Any]]:
        """Get the list of available resources.
        
        Returns:
            List of available resources with their metadata
        """
        return self.available_resources.copy()