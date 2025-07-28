# app.py

import streamlit as st
import os
import ast
import re
import uuid 
import pandas as pd 
import matplotlib.pyplot as plt 

# --- LangChain Imports ---
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.tools import Tool 
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model 
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
# --- FIX --- Import PromptTemplate to help our new tool generate SQL
from langchain_core.prompts import PromptTemplate

# =================================================================================================
# App Configuration
# =================================================================================================
st.set_page_config(page_title="DB Chat Agent", page_icon="📊", layout="wide")
CONVERSATION_HISTORY_LENGTH = 4

if not os.path.exists("plots"):
    os.makedirs("plots")

# =================================================================================================
# Core Agent & Tool Logic
# =================================================================================================

def query_as_list(db, query):
    # (Unchanged)
    try:
        res = db.run(query)
        res = [el for sub in ast.literal_eval(res) for el in sub if el]
        res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
        return list(set(res))
    except Exception:
        return []

# --- FIX --- This new "super-smart" tool replaces the old, unreliable `create_plot` function.
def create_plot_from_question(
    user_question: str, llm, db: SQLDatabase, dialect: str
) -> str:
    """
    This tool takes a user's natural language question, generates a SQL query,
    executes it, verifies the result, creates a plot, and saves it.
    This is the most reliable way to create plots.
    """
    st.write(f"--- Running Smart Plotting Tool for: '{user_question}' ---") # Debugging message

    # Step 1: Use the LLM to generate a SQL query from the user's question.
    template = """
    Based on the user's question, create a single, valid {dialect} SQL query that would return the data needed for a bar chart.
    The query MUST return exactly two columns: the first for the X-axis (labels), and the second for the Y-axis (values).
    Do not add any explanation, preamble, or markdown formatting. Just the raw SQL query.

    Question: {question}
    SQL Query:
    """
    prompt = PromptTemplate(
        input_variables=["question", "dialect"],
        template=template,
    )
    
    sql_query_chain = prompt | llm
    
    try:
        # Generate the SQL query
        sql_query = sql_query_chain.invoke(
            {"question": user_question, "dialect": dialect}
        ).content.strip().replace("```sql", "").replace("```", "") # Clean up LLM output

        st.write(f"🤖 Generated SQL: `{sql_query}`")

        # Step 2: Execute the query
        result_str = db.run(sql_query)
        st.write(f"🔍 Query Result: `{result_str}`")

        # Step 3: Verification (as you requested)
        if not result_str or result_str == "[]":
            return "Verification Error: My generated SQL query returned no data. Please try rephrasing your question."
            
        result_data = ast.literal_eval(result_str)
        
        if not isinstance(result_data, list) or not result_data or not isinstance(result_data[0], tuple) or len(result_data[0]) != 2:
            return f"Verification Error: The SQL query must return two columns. The generated query was: `{sql_query}`. Please try rephrasing."

        # Step 4: Create DataFrame and Plot
        df = pd.DataFrame(result_data, columns=['Label', 'Value'])
        plt.figure(figsize=(10, 6))
        plt.bar(df['Label'], df['Value'])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_filename = f"plots/{uuid.uuid4()}.png"
        plt.savefig(plot_filename)
        plt.close()

        # Step 5: Return a reliable success message
        return f"Plot successfully saved at: {plot_filename}"
            
    except Exception as e:
        plt.close()
        st.error(f"An error occurred in the plotting tool: {e}")
        return f"Execution Error: I failed to create the plot. Please try asking in a different way."

@st.cache_resource(show_spinner="Initializing Agent...")
def get_agent_executor(db_uri, dialect, proper_noun_config):
    """Creates and returns the conversational SQL agent."""
    try:
        db = SQLDatabase.from_uri(db_uri)
        os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.stop()
    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = toolkit.get_tools()
    
    retriever_tool = None
    # (Proper noun logic unchanged)
    
    # --- FIX --- We create our new "super-smart tool" here.
    # It needs access to the llm, db, and dialect, so we use a lambda function.
    plot_super_tool = Tool(
        name="create_plot_from_question",
        func=lambda user_question: create_plot_from_question(
            user_question, llm, db, dialect
        ),
        description="""
        Use this tool ONLY when the user explicitly asks for a plot, chart, graph, or visualization.
        The input to this tool MUST be the user's complete, natural language question.
        This single tool handles everything from creating the SQL query to saving the plot.
        """,
    )
    
    all_tools = sql_tools + ([retriever_tool] if retriever_tool else []) + [plot_super_tool]

    # --- FIX --- The system prompt is now incredibly simple, making the agent's job easy.
    system_message = f"""
    You are an AI assistant. Your job is to decide which tool to use.

    - If the user asks a general question about data, use the `query` tool.
    - If the user asks for a **plot, chart, graph, or visualization**, you MUST use the `create_plot_from_question` tool. Pass the user's entire, unmodified question to this tool.
    - After a plot is created, just confirm it to the user. The UI will add the hyperlink.
    """
    
    agent_executor = create_react_agent(llm, all_tools, messages_modifier=system_message)
    return agent_executor

# =================================================================================================
# Sidebar UI (Unchanged)
# =================================================================================================
with st.sidebar:
    st.header("🧰 Configuration")
    # (The rest of the sidebar is unchanged)
    st.divider()
    db_type = st.selectbox("Select Database Type", ["SQLite", "Databricks"])
    st.divider()
    db_uri = None
    dialect = None
    if db_type == "SQLite":
        st.subheader("📁 SQLite Configuration")
        db_uri_input = st.text_input("Database URI", value="sqlite:///Chinook.db")
        dialect = "sqlite"
        db_uri = db_uri_input
    elif db_type == "Databricks":
        st.subheader("🧱 Databricks Configuration")
        db_host = st.text_input("Server Hostname", help="e.g., adb-....azuredatabricks.net")
        db_path = st.text_input("HTTP Path", help="e.g., /sql/1.0/warehouses/...")
        db_token = st.text_input("Access Token", type="password", help="A Databricks Personal Access Token (PAT).")
        db_catalog = st.text_input("Catalog", help="The catalog to use, e.g., 'hive_metastore'")
        db_schema = st.text_input("Schema (Database)", help="The schema (database) to use, e.g., 'default'")
        dialect = "databricks"
        if all([db_host, db_path, db_token, db_catalog, db_schema]):
            db_uri = f"databricks://token:{db_token}@{db_host}?http_path={db_path}&catalog={db_catalog}&schema={db_schema}"
        else:
            st.warning("Please fill in all Databricks connection details.")
    st.divider()
    with st.expander("🏷️ Advanced: Proper Noun Handling"):
        proper_noun_config = st.text_area("Columns with Proper Nouns", value="Artist.Name\nAlbum.Title\nGenre.Name", height=100)
    st.divider()
    if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
        if db_uri and dialect:
            st.session_state.agent_executor = get_agent_executor(db_uri, dialect, proper_noun_config)
            st.session_state.messages = []
            st.success("Agent is ready!")
            st.rerun()
        else:
            st.error("Database configuration is incomplete. Please check your settings.")

# =================================================================================================
# Main Chat Interface (This logic is correct and remains)
# =================================================================================================
st.title("📊 Chat with Your SQL Database")
st.caption("Connect to SQLite or Databricks, chat with your data, and generate plots.")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

if 'agent_executor' not in st.session_state or st.session_state.agent_executor is None:
    st.info("Please configure your database and initialize the agent from the sidebar to begin.")
else:
    if prompt := st.chat_input("Ask for data or a plot..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 The agent is thinking..."):
                try:
                    history_cutoff = CONVERSATION_HISTORY_LENGTH * 2
                    recent_messages = st.session_state.messages[-history_cutoff:]
                    agent_input_messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in recent_messages]
                    
                    # This logic correctly intercepts the tool message and builds the link.
                    final_answer = ""
                    plot_path = None

                    for step in st.session_state.agent_executor.stream({"messages": agent_input_messages}, stream_mode="values"):
                        last_message = step["messages"][-1]
                        
                        # Intercept the output from our new smart tool
                        if isinstance(last_message, ToolMessage) and "Plot successfully saved at:" in last_message.content:
                            path_match = re.search(r"plots/[-a-zA-Z0-9]+\.png", last_message.content)
                            if path_match:
                                plot_path = path_match.group(0)

                        elif isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls:
                            final_answer = last_message.content
                    
                    if plot_path:
                        display_message = f"{final_answer}\n\n[Click here to view the plot]({plot_path})"
                    else:
                        # If the smart tool failed, its error message will be in the final_answer
                        display_message = final_answer

                    st.markdown(display_message, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": display_message})

                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})