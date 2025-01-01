import streamlit as st
from sqlalchemy import create_engine, inspect, text
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.chat_models import ChatOpenAI
import os
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import json

# Set wide layout
st.set_page_config(layout="wide", page_title="KarmaOpsAI", page_icon=":bar_chart:")

from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg2://postgres:delusional@localhost/postgres",
    pool_size=10,  # Maximum number of connections in the pool
    max_overflow=5,  # Additional connections to open if the pool is full
    pool_timeout=30,  # Wait time before a timeout error
)

# Environment variables for database and OpenAI API key
os.environ["OPENAI_API_KEY"] = ''
db_user = os.getenv("DB_USER", "postgres")
db_password = os.getenv("DB_PASSWORD", "delusional")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME", "postgres")

# Connect to PostgreSQL using SQLAlchemy
engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
db = SQLDatabase(engine)

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Function to create a contextual prompt
def create_contextual_prompt(question, tables_info, contextual_memory):
    context = "The database has the following tables and their columns:\n"
    for schema, tables in tables_info.items():
        for table, columns in tables.items():
            context += f"Schema: {schema}, Table: {table}, Columns: {', '.join(columns)}\n"

    if contextual_memory:
        context += "\nPrevious queries and answers:\n"
        for i, memory in enumerate(contextual_memory):
            context += f"Query {i + 1}: {memory['question']}\nAnswer {i + 1}: {memory['answer']}\n"

    context = (
        "You are a veteran in creating queries for SQL in Postgres. "
        f"Question: {question}\n"
        "Generate an accurate SQL query based on the following important calculation rules and definitions:\n\n"
        "Important calculation rules and definitions:\n"
        "1. Utilization is calculated as Gallons delivered divided by Tank Size. Delivered volume should be taken from "
        "the `delivery_report` table. Avoid repetitive customer names and ignore null values.\n"
        "2. 'Based on last fill' means the last time total gallons were delivered by tank size.\n"
        "3. Overall Utilization refers to the average total fill in the lifetime of the tank at the customer location "
        "by tank size.\n"
        "4. Last 5 fill average is the average of the last 5 fills of gallons delivered by tank size at the customer "
        "location.\n "
        "Identify the worst assets as those with the lowest utilization percentages in each category. The worst "
        "assets can "
        "be calculated based on last fill utilization, last 5 fill utilization, or overall utilization according to "
        "the prompt.\n "
        "5. For asset types:\n"
        "   - Rental Assets: Assets marked as 'rental' in the `asset_type` column.\n"
        "   - Customer Assets: Assets marked as 'customer-owned'.\n"
        "   - Delivery Assets: Assets related to delivery operations.\n"
        "6. When calculating days between fills, use the `delivery_date` column.\n"
        "7. For customer locations, use the `location` table.\n"
        "8. For delivery volume buckets, use the ranges: 1-100, 100-500, 500-1000, 1000-5000, and 5000-10000 gallons.\n"
        "9. If 'State' is requested, extract it from the `address` column using string manipulation.\n"
        "10. For customer-related questions, refer to the cleaned customer locations table. Apply necessary join conditions "
        "    and use data from the cleaned customers list if required.\n"
        "11. Dry runs refer to unsuccessful fills where a driver attempted delivery but no fuel was delivered.\n"
        "12. Do not use `LIMIT 5` in queries unless explicitly mentioned.\n"
        "13. Avoid using aggregate functions (e.g., `AVG`) directly with window functions (e.g., `LAG`). Instead, calculate "
        "    window function results in a subquery or CTE and apply the aggregate function in the outer query.\n"
        "14. Ignore null values unless they are crucial for the analysis.\n"
        "15. Basic filters for data analysis include:\n"
        "    - **Date Range:** Filter data by daily, weekly, monthly, or custom date ranges.\n"
        "    - **By Hubs:** Filter data by specific operational hubs or regions. Get the hub name from the `hubs` table by "
        "      creating an indirect join between the `delivery_report` and `hubs` tables.\n"
        "    - **By Drivers:** Filter data by specific driver IDs or names.\n"
        "    - **By Customers:** Filter by specific customers or groups of customers.\n"
        "    - **By Customer and Customer Locations:** Analyze data at the customer level and their respective locations.\n"
        "    - **Volume Filters:** Use predefined delivery volume ranges (e.g., 1-100, 100-500 gallons).\n"
        "    - **Status Filters:** Include or exclude records based on statuses like completed, pending, or canceled.\n"
        "    - **Utilization Filters:** Narrow down records where utilization falls below a specified threshold.\n"
        "    - **Null Handling:** Include or exclude null values based on the analysis requirements.\n"
        "16. Enhance SQL queries by:\n"
        "    - Using subqueries or Common Table Expressions (CTEs) to preprocess data in intermediary tables. "
        "      Verify the accuracy of the joins before aggregating data.\n"
        "    - Ensuring grouping is applied after all necessary joins to maintain accurate aggregated data.\n"
        "17. For comparative analysis from the `delivery_report`:\n"
        "    - **Total Stops:** Count the distinct `delivery_id` values per date.\n"
        "    - **Total Gallons:** Sum the `volume` per date.\n"
        "    - **Time Periods:**\n"
        "        - Weekly: Compare current (last 7 days) vs. previous (7 days before that).\n"
        "        - Monthly: Compare the current calendar month vs. the previous calendar month.\n"
        "        - Quarterly: Compare the current calendar quarter vs. the previous calendar quarter.\n"
    )
    return context


# Get tables and their columns
@st.cache_data
def get_tables_info():
    inspector = inspect(engine)
    schemas = inspector.get_schema_names()
    tables_info = {}
    for schema in schemas:
        if schema in ['pg_catalog', 'information_schema', 'pg_toast']:
            continue
        tables = inspector.get_table_names(schema=schema)
        if tables:
            for table in tables:
                columns = [col['name'] for col in inspector.get_columns(table, schema=schema)]
                if schema not in tables_info:
                    tables_info[schema] = {}
                tables_info[schema][table] = columns
    return tables_info


tables_info = get_tables_info()

# Initialize session state
if "memory" not in st.session_state:
    st.session_state.memory = []  # List to store prompts and results
if "show_memory" not in st.session_state:
    st.session_state.show_memory = False  # Memory display toggle
if "contextual_memory" not in st.session_state:
    st.session_state.contextual_memory = []  # List to store query-answer pairs for context

# Streamlit app
st.title("KarmaOpsAI: Conversational Insights")

# Create two columns for layout
col1, col2 = st.columns([10, 1])

# Column 2: Memory toggle
with col2:
    if st.button("📝 Memory"):
        st.session_state.show_memory = not st.session_state.show_memory

# Column 1: Input and query results
with col1:
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question:", "")

    if question:
        with st.spinner("Generating SQL query and fetching results..."):
            # Create prompt with contextual memory
            prompt = create_contextual_prompt(question, tables_info, st.session_state.contextual_memory)

            try:
                # Generate SQL query using LangChain
                generate_query = create_sql_query_chain(llm, db)
                query_result = generate_query.invoke({"question": prompt})

                if isinstance(query_result, str):
                    sql_query = query_result
                else:
                    raise ValueError(f"Unexpected query result format: {query_result}")

                # Execute the SQL query
                with engine.connect() as connection:
                    result = connection.execute(text(sql_query))
                    rows = result.fetchall()
                    columns = result.keys()

                    # Convert rows to JSON
                    json_result = [dict(zip(columns, row)) for row in rows]

                    # Store the query data in the `training_data` table
                    insert_query = text("""
                        INSERT INTO public.training_data (user_query, generated_sql, query_results)
                        VALUES (:user_query, :generated_sql, :query_results)
                    """)
                    try:
                        connection.execute(insert_query, {
                            "user_query": question,
                            "generated_sql": sql_query,
                            "query_results": json.dumps(json_result)  # Convert results to JSON
                        })
                        connection.commit()  # Explicitly commit the transaction
                        st.success("Data saved to training_data table!")
                    except Exception as e:
                        st.error(f"Failed to save data to training_data table: {e}")

                    # Display the result
                    st.success("Query executed successfully!")
                    if json_result:
                        df = pd.DataFrame(json_result)

                        # Display table and visualization side by side
                        table_col, graph_col = st.columns(2)

                        with table_col:
                            st.subheader("Query Results")
                            st.dataframe(df)

                        with graph_col:
                            st.subheader("Visualize Data")
                            graph_type = st.selectbox("Select graph type:",
                                                      ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart",
                                                       "Boxplot", "Heatmap", "Area Chart", "Histogram"])

                            default_x = df.columns[0] if len(df.columns) > 0 else None
                            default_y = df.columns[1] if len(df.columns) > 1 else None

                            if graph_type == "Bar Chart":
                                fig = px.bar(df, x=default_x, y=default_y, title="Bar Chart")
                            elif graph_type == "Line Chart":
                                fig = px.line(df, x=default_x, y=default_y, title="Line Chart")
                            elif graph_type == "Scatter Plot":
                                fig = px.scatter(df, x=default_x, y=default_y, title="Scatter Plot")
                            elif graph_type == "Pie Chart":
                                fig = px.pie(df, names=default_x, values=default_y, title="Pie Chart")
                            elif graph_type == "Boxplot":
                                fig = px.box(df, x=default_x, y=default_y, title="Boxplot")
                            elif graph_type == "Heatmap":
                                corr_matrix = df.select_dtypes(include='number').corr()
                                fig = ff.create_annotated_heatmap(z=corr_matrix.values, x=corr_matrix.columns.values,
                                                                  y=corr_matrix.index.values, colorscale="Viridis")
                            elif graph_type == "Area Chart":
                                fig = px.area(df, x=default_x, y=default_y, title="Area Chart")
                            elif graph_type == "Histogram":
                                fig = px.histogram(df, x=default_x, title="Histogram")

                            st.plotly_chart(fig)

                        # Append to memory
                        st.session_state.memory.append({
                            "question": question,
                            "dataframe": df,
                            "graph": fig,
                            "graph_type": graph_type,
                            "answer": json_result
                        })
                    else:
                        st.warning("No results found for the query.")

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Toggle memory section visibility
if st.session_state.show_memory:
    st.subheader("Session Memory")
    for i, item in enumerate(st.session_state.memory):
        st.markdown(f"### Query {i + 1}: {item['question']}")
        memory_table_col, memory_graph_col = st.columns(2)

        with memory_table_col:
            st.subheader(f"Query {i + 1} Results")
            st.dataframe(item['dataframe'])

        with memory_graph_col:
            st.subheader(f"Query {i + 1} Visualization")
            st.plotly_chart(item['graph'], key=f"graph_{i}")
