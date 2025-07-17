# MCP (Model Context Protocol) Integration

This document describes the integration between LangChain and MCP (Model Context Protocol), enabling LangChain applications to interact with MCP servers.

## Overview

The Model Context Protocol (MCP) is a standard developed by Anthropic that enables AI models to securely access external data and tools through a unified interface. LangChain's MCP integration provides tools that can connect to MCP servers and execute their exposed functionality.

## Components

### MCPTool

`MCPTool` is a LangChain tool that enables execution of tools exposed by MCP servers.

```python
from langchain_community.tools.mcp.tool import MCPTool

# Connect to an MCP server
mcp_tool = MCPTool(server_url="http://localhost:8000")

# The tool automatically discovers available MCP tools
print(f"Available tools: {mcp_tool.get_available_tools()}")
```

**Features:**
- Automatic tool discovery from MCP servers
- JSON-based tool calling interface
- Error handling and validation
- Async support

**Usage:**
```python
import json

# Call an MCP tool
input_data = json.dumps({
    "tool_name": "file_read",
    "arguments": {"filename": "example.txt"}
})

result = mcp_tool._run(input_data)
```

### MCPResourceTool

`MCPResourceTool` provides access to resources exposed by MCP servers.

```python
from langchain_community.tools.mcp.tool import MCPResourceTool

# Connect to an MCP server for resource access
resource_tool = MCPResourceTool(server_url="http://localhost:8000")

# Access a resource by URI
result = resource_tool._run("file://documents/readme.txt")
```

**Features:**
- Resource discovery and listing
- URI-based resource access
- Support for various resource types (files, databases, APIs)

### MCPClient

`MCPClient` is the underlying client that communicates with MCP servers.

```python
from langchain_community.tools.mcp.client import MCPClient

client = MCPClient(server_url="http://localhost:8000")

# List available tools
tools = client.list_tools()

# Call a tool
result = client.call_tool("calculator", {"expression": "2 + 2"})

# List and access resources
resources = client.list_resources()
content = client.read_resource("file://data/config.json")
```

## Usage in LangChain Applications

### With Agents

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools.mcp.tool import MCPTool
from langchain_core.prompts import PromptTemplate

# Create MCP tools
mcp_tool = MCPTool(server_url="http://localhost:8000")

# Use with an agent
tools = [mcp_tool]
# ... create and use agent with tools
```

### Standalone Usage

```python
from langchain_community.tools.mcp.tool import MCPTool, MCPResourceTool

# Create tools
mcp_tool = MCPTool(server_url="http://localhost:8000")
resource_tool = MCPResourceTool(server_url="http://localhost:8000")

# Use directly
result = mcp_tool._run('{"tool_name": "echo", "arguments": {"message": "Hello"}}')
content = resource_tool._run("file://documents/readme.txt")
```

## Configuration

### Server URL

The primary configuration is the MCP server URL:

```python
# Local server
mcp_tool = MCPTool(server_url="http://localhost:8000")

# Remote server
mcp_tool = MCPTool(server_url="https://mcp.example.com")
```

### Timeout

Configure request timeout:

```python
from langchain_community.tools.mcp.client import MCPClient

client = MCPClient(
    server_url="http://localhost:8000",
    timeout=60  # 60 seconds
)
```

## MCP Protocol Details

The integration follows the MCP specification:

### Tool Calling
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": {}
  }
}
```

### Resource Access
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/read",
  "params": {
    "uri": "resource_uri"
  }
}
```

## Error Handling

The MCP integration includes comprehensive error handling:

- **Connection errors**: Gracefully handled with informative error messages
- **Tool not found**: Clear indication of available tools
- **Invalid input**: JSON validation and error reporting
- **Server errors**: Proper propagation of MCP server error responses

## Examples

### File Operations
```python
import json
from langchain_community.tools.mcp.tool import MCPTool

mcp_tool = MCPTool(server_url="http://localhost:8000")

# Read a file
read_input = json.dumps({
    "tool_name": "file_read",
    "arguments": {"path": "/path/to/file.txt"}
})
content = mcp_tool._run(read_input)

# Write a file
write_input = json.dumps({
    "tool_name": "file_write",
    "arguments": {
        "path": "/path/to/output.txt",
        "content": "Hello, MCP!"
    }
})
result = mcp_tool._run(write_input)
```

### Database Queries
```python
# Execute a database query via MCP
query_input = json.dumps({
    "tool_name": "db_query",
    "arguments": {
        "query": "SELECT * FROM users WHERE active = true",
        "database": "production"
    }
})
results = mcp_tool._run(query_input)
```

### API Calls
```python
# Make API calls through MCP
api_input = json.dumps({
    "tool_name": "http_request",
    "arguments": {
        "method": "GET",
        "url": "https://api.example.com/data",
        "headers": {"Authorization": "Bearer token"}
    }
})
response = mcp_tool._run(api_input)
```

## Best Practices

1. **Server Health**: Always check if the MCP server is running before using tools
2. **Error Handling**: Implement proper error handling for network and server issues
3. **Input Validation**: Validate JSON input before calling tools
4. **Resource Management**: Use context managers for client connections when possible
5. **Security**: Ensure MCP servers are properly secured and authenticated

## Troubleshooting

### Common Issues

1. **Connection Refused**: Check if MCP server is running and accessible
2. **Tool Not Found**: Verify tool names with `get_available_tools()`
3. **Invalid JSON**: Ensure input is properly formatted JSON
4. **Timeout**: Increase timeout for long-running operations

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- Authentication support for secure MCP servers
- Prompt template integration
- Batch operations support
- WebSocket connections for real-time communication
- Enhanced caching mechanisms

## Resources

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP GitHub Repository](https://github.com/modelcontextprotocol)
- [LangChain Tools Documentation](https://python.langchain.com/docs/concepts/tools/)