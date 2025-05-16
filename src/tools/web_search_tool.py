from google.adk.core import Tool
import time # For simulating delay

class WebSearchTool(Tool):
    """
    A tool for performing web searches. 
    Currently returns mock results. Future implementation will use a real search API.
    """
    def __init__(self, tool_id: str = "web_search_tool"):
        """
        Initializes the WebSearchTool.

        Args:
            tool_id (str): The unique identifier for this tool instance.
        """
        super().__init__(tool_id=tool_id)
        # API key for a real search service would be initialized here, potentially from .env
        # Example: self.search_api_key = os.getenv("SEARCH_API_KEY")

    def execute(self, session_state: dict, query: str, num_results: int = 3, **kwargs) -> dict:
        """
        Executes the web search for the given query.

        Args:
            session_state (dict): The current session state (not directly used by mock tool yet).
            query (str): The search query.
            num_results (int): The desired number of search results.
            **kwargs: Additional arguments (not used by mock tool yet).

        Returns:
            dict: A dictionary containing the search results under the key 'search_results'.
                  Each result is a dict with 'title', 'link', and 'snippet'.
        """
        print(f"Tool '{self.tool_id}': Executing search for query: '{query}' (max {num_results} results).")
        
        # Simulate network delay for a more realistic mock
        time.sleep(0.5)

        # Mock search results
        mock_results = [
            {
                "title": f"Mock Result 1 for '{query}'",
                "link": f"https://example.com/search?q={query.replace(' ', '+')}&result=1",
                "snippet": f"This is the first mock search result snippet related to '{query}'. It provides some basic information."
            },
            {
                "title": f"Mock Result 2: All about '{query}'",
                "link": f"https://example.com/info/{query.replace(' ', '_')}",
                "snippet": f"Detailed information and resources about '{query}' can be found here. This is the second mock snippet."
            },
            {
                "title": f"'{query}' - A Comprehensive Guide (Mock)",
                "link": f"https://guides.example.com/{query.replace(' ', '-')}",
                "snippet": f"The third mock result offers a comprehensive guide to understanding '{query}'. Learn more now!"
            },
            {
                "title": f"Exploring the Nuances of '{query}' (Mock Data)",
                "link": f"https://example.org/explore?topic={query.replace(' ', '%20')}",
                "snippet": f"A deeper dive into '{query}', presenting various perspectives and mock data points for consideration."
            }
        ]

        results_to_return = mock_results[:num_results]
        print(f"Tool '{self.tool_id}': Returning {len(results_to_return)} mock results.")

        # The ADK expects tools to return a dictionary that can be merged into session_state.
        # The calling agent will typically process this dictionary.
        return {
            "search_query": query,
            "search_results": results_to_return
        }

if __name__ == '__main__':
    print("Testing WebSearchTool...")
    search_tool = WebSearchTool()

    # Test case 1
    query1 = "benefits of renewable energy"
    print(f"\n--- Test Case 1: Query: '{query1}' ---")
    results1 = search_tool.execute(session_state={}, query=query1, num_results=2)
    print("Results:")
    for i, res in enumerate(results1.get("search_results", [])):
        print(f"  {i+1}. Title: {res['title']}")
        print(f"     Link: {res['link']}")
        print(f"     Snippet: {res['snippet']}")

    # Test case 2
    query2 = "future of AI in healthcare"
    print(f"\n--- Test Case 2: Query: '{query2}' (default num_results) ---")
    results2 = search_tool.execute(session_state={}, query=query2)
    print("Results:")
    for i, res in enumerate(results2.get("search_results", [])):
        print(f"  {i+1}. Title: {res['title']}")
        print(f"     Link: {res['link']}")
        print(f"     Snippet: {res['snippet']}")
    
    # Test case 3: Requesting more results than available in mock
    query3 = "latest Mars rover findings"
    print(f"\n--- Test Case 3: Query: '{query3}' (num_results=5) ---")
    results3 = search_tool.execute(session_state={}, query=query3, num_results=5)
    print("Results:")
    for i, res in enumerate(results3.get("search_results", [])):
        print(f"  {i+1}. Title: {res['title']}")
        print(f"     Link: {res['link']}")
        print(f"     Snippet: {res['snippet']}")

    print("\nWebSearchTool testing finished.")
