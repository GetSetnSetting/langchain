"""Test MCP tool integration with LangChain."""

import json
import pytest
from unittest.mock import Mock, patch

from langchain_community.tools.mcp.tool import MCPTool, MCPResourceTool
from langchain_community.tools.mcp.client import MCPClient, MCPTool as MCPToolSchema


class TestMCPTool:
    """Test cases for MCPTool."""
    
    @patch('langchain_community.tools.mcp.client.httpx.Client')
    def test_mcp_tool_initialization(self, mock_httpx_client):
        """Test MCP tool initialization with mocked server."""
        # Mock the HTTP client
        mock_client = Mock()
        mock_httpx_client.return_value = mock_client
        
        # Mock server response for list_tools
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": {
                "tools": [
                    {
                        "name": "test_tool",
                        "description": "A test tool",
                        "input_schema": {"type": "object", "properties": {}}
                    }
                ]
            }
        }
        mock_client.post.return_value = mock_response
        
        # Initialize MCP tool
        tool = MCPTool(server_url="http://localhost:8000")
        
        # Verify initialization
        assert tool.name == "mcp_tool"
        assert len(tool.available_tools) == 1
        assert tool.available_tools[0]["name"] == "test_tool"
        assert "test_tool: A test tool" in tool.description
    
    @patch('langchain_community.tools.mcp.client.httpx.Client')
    def test_mcp_tool_execution(self, mock_httpx_client):
        """Test MCP tool execution."""
        # Mock the HTTP client
        mock_client = Mock()
        mock_httpx_client.return_value = mock_client
        
        # Mock server responses
        def mock_post(url, **kwargs):
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            
            if "tools/list" in url:
                mock_response.json.return_value = {
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "description": "Echo input back",
                                "input_schema": {"type": "object", "properties": {"message": {"type": "string"}}}
                            }
                        ]
                    }
                }
            elif "tools/call" in url:
                mock_response.json.return_value = {
                    "result": {"output": "Hello, World!"}
                }
            
            return mock_response
        
        mock_client.post.side_effect = mock_post
        
        # Initialize and test tool
        tool = MCPTool(server_url="http://localhost:8000")
        
        # Test valid tool call
        input_json = json.dumps({
            "tool_name": "echo",
            "arguments": {"message": "Hello, World!"}
        })
        
        result = tool._run(input_json)
        
        # Verify result
        assert "Hello, World!" in result
        
        # Test invalid tool call
        invalid_input = json.dumps({
            "tool_name": "nonexistent_tool",
            "arguments": {}
        })
        
        result = tool._run(invalid_input)
        assert "Error: Tool 'nonexistent_tool' not found" in result
    
    @patch('langchain_community.tools.mcp.client.httpx.Client')
    def test_mcp_tool_invalid_input(self, mock_httpx_client):
        """Test MCP tool with invalid input."""
        # Mock the HTTP client
        mock_client = Mock()
        mock_httpx_client.return_value = mock_client
        
        # Mock empty tools response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"result": {"tools": []}}
        mock_client.post.return_value = mock_response
        
        # Initialize tool
        tool = MCPTool(server_url="http://localhost:8000")
        
        # Test invalid JSON
        result = tool._run("invalid json")
        assert "Input validation error" in result
        
        # Test missing tool_name
        result = tool._run('{"arguments": {}}')
        assert "Input validation error" in result


class TestMCPResourceTool:
    """Test cases for MCPResourceTool."""
    
    @patch('langchain_community.tools.mcp.client.httpx.Client')
    def test_mcp_resource_tool_initialization(self, mock_httpx_client):
        """Test MCP resource tool initialization."""
        # Mock the HTTP client
        mock_client = Mock()
        mock_httpx_client.return_value = mock_client
        
        # Mock server response for list_resources
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": {
                "resources": [
                    {
                        "uri": "file://test.txt",
                        "name": "test.txt",
                        "mime_type": "text/plain",
                        "description": "A test file"
                    }
                ]
            }
        }
        mock_client.post.return_value = mock_response
        
        # Initialize MCP resource tool
        tool = MCPResourceTool(server_url="http://localhost:8000")
        
        # Verify initialization
        assert tool.name == "mcp_resource"
        assert len(tool.available_resources) == 1
        assert tool.available_resources[0]["uri"] == "file://test.txt"
        assert "file://test.txt" in tool.description


class TestMCPClient:
    """Test cases for MCPClient."""
    
    @patch('langchain_community.tools.mcp.client.httpx.Client')
    def test_mcp_client_list_tools(self, mock_httpx_client):
        """Test MCP client tool listing."""
        # Mock the HTTP client
        mock_client = Mock()
        mock_httpx_client.return_value = mock_client
        
        # Mock server response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": {
                "tools": [
                    {
                        "name": "calculator",
                        "description": "Perform calculations",
                        "input_schema": {"type": "object"}
                    }
                ]
            }
        }
        mock_client.post.return_value = mock_response
        
        # Test client
        client = MCPClient(server_url="http://localhost:8000")
        tools = client.list_tools()
        
        # Verify results
        assert len(tools) == 1
        assert tools[0].name == "calculator"
        assert tools[0].description == "Perform calculations"
    
    @patch('langchain_community.tools.mcp.client.httpx.Client')
    def test_mcp_client_call_tool(self, mock_httpx_client):
        """Test MCP client tool calling."""
        # Mock the HTTP client
        mock_client = Mock()
        mock_httpx_client.return_value = mock_client
        
        # Mock server response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": {"answer": 42}
        }
        mock_client.post.return_value = mock_response
        
        # Test client
        client = MCPClient(server_url="http://localhost:8000")
        result = client.call_tool("calculator", {"expression": "6*7"})
        
        # Verify results
        assert result["answer"] == 42


def test_mcp_tool_import():
    """Test that MCP tools can be imported from langchain_community.tools."""
    from langchain_community.tools import MCPTool, MCPResourceTool
    
    # Verify classes are available
    assert MCPTool is not None
    assert MCPResourceTool is not None