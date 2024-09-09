# Import required classes and functions from langchain module
from langchain_experimental.agents import create_pandas_dataframe_agent
from typing import Any
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI, OpenAIEmbeddings
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS
import os
from langchain.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from datetime import datetime
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional
from langchain.chains import create_structured_output_runnable

def create_dataframe_agent(df: Any):
    """
    Create an agent for processing pandas DataFrame.

    Args:
        df: Pandas DataFrame to process.

    Returns:
        Pandas dataframe agent.
    """
    # Create an OpenAI object.
    proxy_client = get_proxy_client('gen-ai-hub')
    llm = ChatOpenAI(proxy_model_name='gpt-35-turbo-16k', proxy_client=proxy_client)

    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type="openai-tools",
        max_iterations=30,
        number_of_head_rows=len(df.columns)
    )

def prompt_similarity_search(prompt, prompt_df):
    """
    Search for similar prompts based on a given prompt and a dataframe of prompts.

    Args:
        prompt: Query prompt.
        prompt_df: DataFrame containing prompts and corresponding responses.

    Returns:
        Formatted prompt with similar examples.
    """
    # Create an OpenAI object.
    proxy_client = get_proxy_client('gen-ai-hub')
    embedding_llm = OpenAIEmbeddings(proxy_model_name='text-embedding-ada-002', proxy_client=proxy_client)

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="input: {input}\n output: {output}",
    )

    examples = [{"input": row['Question'], "output": row['Answer']} for index, row in prompt_df.iterrows()]

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        embedding_llm,
        FAISS,
        k=1,
    )

    fewshots = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=f"Generate the output inline to the input. Don't explain code if there are any. Today {datetime.now()}",
        suffix="input: {query}\n output:",
        input_variables=["query"],
    )

    fewshotprompt = "'''" + os.linesep + fewshots.format(query=prompt) + os.linesep + "'''"
    return fewshotprompt

def query_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """
    now = f"Today is {datetime.now()}"
    prompt = (
        """
        For the following query, 
        Generate the output inline to the input. {now}

        Below is the query.
        Query: 
        """
        + query
    )

    # Run the prompt through the agent.
    print(prompt)
    response = agent.run(prompt)
    print(response.__str__())
    # Convert the response to a string.
    return response

class Promotion(BaseModel):
    '''Identifying information about promotions to be created/updated/deleted.'''

    logicalSystem: str = Field(..., description="ID of the promotion")
    product_id: str = Field(..., description="ID of the product")
    promotionName: str = Field(..., description="name of the product name")
    promotion_type: str = Field(..., description="type of the promotion")
    promoDescription: Optional[str] = Field(None, description="promotion description text")
    businessUnitID: str = Field(..., description="Business unit")
    businessUnitType: str = Field(..., description="Business unit type")
    location: str = Field(..., description="Business unit location")
    startDate: str = Field(..., description="Start date of promotion")
    activeDate: str = Field(..., description="Start date of promotion")
    endDate: str = Field(..., description="Expired date of promotion")
    createdOn: str = Field(..., description="Current date")
    changedOn: str = Field(..., description="Current date")

class SalesGraph(BaseModel):
    '''Identifying information about sales graph powered by plotly'''

    graph_name: str = Field(..., description="Name of graph")
    graph_type: str = Field(..., description="Type of the graph to be plotted")
    x_field_name: str = Field(..., description="x coordinate for plotly graph")
    x_field_selected_values: str  = Field(..., description="selected values for x coordinate to be plotted")
    x_field_filter_from: str  = Field(..., description="from filter value for x coordinate to be plotted")
    x_field_filter_to: str  = Field(..., description="to filter value for x coordinate to be plotted")
    x_field_filter_by: str  = Field(..., description="filtered by value for x coordinate to be plotted")
    category: str = Field(..., description="category name of product")
    product_name: str = Field(..., description="name of the product")

    y_field_name: str = Field(..., description="y coordinate for plotly graph")
    color: str = Field(..., description="plotly color field name")

def create_structured_output_agent(output_schema):
    """
    Create an agent for processing structured output based on the given schema.

    Args:
        output_schema: Schema for structured output.

    Returns:
        Structured output agent.
    """
    # Create an OpenAI object.
    proxy_client = get_proxy_client('gen-ai-hub')
    llm = ChatOpenAI(proxy_model_name='gpt-35-turbo-16k', proxy_client=proxy_client, temperature=0)

    return create_structured_output_runnable(
        output_schema,
        llm,
        mode="openai-functions",
        enforce_function_usage=True,
        return_single=True
    )