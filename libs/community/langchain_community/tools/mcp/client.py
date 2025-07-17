"""Basic MCP client implementation for LangChain integration."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MCPTool(BaseModel):
    """Represents an MCP tool."""
    
    name: str = Field(description="The name of the tool")
    description: str = Field(description="Description of what the tool does")
    input_schema: Dict[str, Any] = Field(description="JSON schema for tool inputs")


class MCPResource(BaseModel):
    """Represents an MCP resource."""
    
    uri: str = Field(description="URI of the resource")
    name: str = Field(description="Name of the resource")
    mime_type: Optional[str] = Field(default=None, description="MIME type of the resource")
    description: Optional[str] = Field(default=None, description="Description of the resource")


class MCPPrompt(BaseModel):
    """Represents an MCP prompt template."""
    
    name: str = Field(description="Name of the prompt")
    description: str = Field(description="Description of the prompt")
    arguments: List[Dict[str, Any]] = Field(default=[], description="Prompt arguments")


class MCPClient:
    """Basic MCP client for communicating with MCP servers."""
    
    def __init__(self, server_url: str, timeout: int = 30):
        """Initialize the MCP client.
        
        Args:
            server_url: URL of the MCP server
            timeout: Timeout for requests in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self._tools_cache: Optional[List[MCPTool]] = None
        self._resources_cache: Optional[List[MCPResource]] = None
        self._prompts_cache: Optional[List[MCPPrompt]] = None
    
    def __del__(self):
        """Cleanup the HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
    
    def list_tools(self, force_refresh: bool = False) -> List[MCPTool]:
        """List available tools from the MCP server.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            List of available MCP tools
        """
        if self._tools_cache is None or force_refresh:
            try:
                response = self.client.post(
                    f"{self.server_url}/tools/list",
                    headers={"Content-Type": "application/json"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
                )
                response.raise_for_status()
                data = response.json()
                
                if "result" in data and "tools" in data["result"]:
                    self._tools_cache = [
                        MCPTool(**tool) for tool in data["result"]["tools"]
                    ]
                else:
                    logger.warning("Unexpected response format from MCP server")
                    self._tools_cache = []
                    
            except Exception as e:
                logger.error(f"Failed to list tools from MCP server: {e}")
                self._tools_cache = []
        
        return self._tools_cache or []
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Result from the tool execution
        """
        try:
            response = self.client.post(
                f"{self.server_url}/tools/call",
                headers={"Content-Type": "application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            
            if "result" in data:
                return data["result"]
            elif "error" in data:
                raise RuntimeError(f"MCP tool error: {data['error']}")
            else:
                raise RuntimeError("Unexpected response format from MCP server")
                
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name}: {e}")
            raise
    
    def list_resources(self, force_refresh: bool = False) -> List[MCPResource]:
        """List available resources from the MCP server.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            List of available MCP resources
        """
        if self._resources_cache is None or force_refresh:
            try:
                response = self.client.post(
                    f"{self.server_url}/resources/list",
                    headers={"Content-Type": "application/json"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "resources/list"}
                )
                response.raise_for_status()
                data = response.json()
                
                if "result" in data and "resources" in data["result"]:
                    self._resources_cache = [
                        MCPResource(**resource) for resource in data["result"]["resources"]
                    ]
                else:
                    logger.warning("Unexpected response format from MCP server")
                    self._resources_cache = []
                    
            except Exception as e:
                logger.error(f"Failed to list resources from MCP server: {e}")
                self._resources_cache = []
        
        return self._resources_cache or []
    
    def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource from the MCP server.
        
        Args:
            uri: URI of the resource to read
            
        Returns:
            Resource content
        """
        try:
            response = self.client.post(
                f"{self.server_url}/resources/read",
                headers={"Content-Type": "application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "resources/read",
                    "params": {"uri": uri}
                }
            )
            response.raise_for_status()
            data = response.json()
            
            if "result" in data:
                return data["result"]
            elif "error" in data:
                raise RuntimeError(f"MCP resource error: {data['error']}")
            else:
                raise RuntimeError("Unexpected response format from MCP server")
                
        except Exception as e:
            logger.error(f"Failed to read resource {uri}: {e}")
            raise
    
    def list_prompts(self, force_refresh: bool = False) -> List[MCPPrompt]:
        """List available prompts from the MCP server.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            List of available MCP prompts
        """
        if self._prompts_cache is None or force_refresh:
            try:
                response = self.client.post(
                    f"{self.server_url}/prompts/list",
                    headers={"Content-Type": "application/json"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "prompts/list"}
                )
                response.raise_for_status()
                data = response.json()
                
                if "result" in data and "prompts" in data["result"]:
                    self._prompts_cache = [
                        MCPPrompt(**prompt) for prompt in data["result"]["prompts"]
                    ]
                else:
                    logger.warning("Unexpected response format from MCP server")
                    self._prompts_cache = []
                    
            except Exception as e:
                logger.error(f"Failed to list prompts from MCP server: {e}")
                self._prompts_cache = []
        
        return self._prompts_cache or []
    
    def get_prompt(self, prompt_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get a prompt from the MCP server.
        
        Args:
            prompt_name: Name of the prompt to get
            arguments: Arguments to pass to the prompt
            
        Returns:
            Prompt content
        """
        try:
            params = {"name": prompt_name}
            if arguments:
                params["arguments"] = arguments
                
            response = self.client.post(
                f"{self.server_url}/prompts/get",
                headers={"Content-Type": "application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "prompts/get",
                    "params": params
                }
            )
            response.raise_for_status()
            data = response.json()
            
            if "result" in data:
                return data["result"]
            elif "error" in data:
                raise RuntimeError(f"MCP prompt error: {data['error']}")
            else:
                raise RuntimeError("Unexpected response format from MCP server")
                
        except Exception as e:
            logger.error(f"Failed to get prompt {prompt_name}: {e}")
            raise