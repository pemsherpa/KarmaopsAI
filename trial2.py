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

# Environment variables for database and OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
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

    context +="""
        Important rules and definitions for calculations:
        1. Utilization = (Gallons Delivered / Tank Size) * 100%.
        2. Last Fill Utilization = Utilization calculated for the last delivery at a customer location.
        3. Overall Utilization = Average of utilization over all deliveries in a tank's lifetime at a customer location.
        4. Last 5 Fill Utilization = Average utilization for the last 5 deliveries at a customer location.
        5. Asset Classification:
           - Rental Assets: Identified by 'rental' in the asset_type column.
           - Customer Assets: Identified by 'customer-owned' in the asset_type column.
           - Delivery Assets: Related to operations, defined as needed.
        6. Delivery Volume Buckets: Predefined ranges: 1-100, 100-500, 500-1000, 1000-5000, 5000-10000 gallons.
        7. Use ship to erp id to join customers, locations, and assets.
        8. Dry runs = Unsuccessful fills where Gallons Delivered = 0.
        9. Filters to apply dynamically:
           - Date Range: Daily, weekly, monthly, custom.
           - Hubs, Drivers, Dispatchers, Customers, Locations: Filter data at multiple levels.
        10. SQL best practices:
           - Proper JOINs for relationships.
           - Handle NULL values appropriately.
           - CASE statements for buckets/ranges.
           - Use subqueries for window functions, aggregate calculations.
           - Avoid LIMIT unless specified.
        11. Ignore null values if not crucial.
        12. Basic filters refer to the criteria applied to narrow down data for analysis:
            - *Date Range:* Apply filters to analyze data by daily, weekly, monthly, or custom date ranges.
            - *By Hubs:* Filter data by specific operational hubs or regions.Get Hub's name from Hubs table for that try to create an indirect join of delivery report and hubs.
            - *By Drivers:* Select records pertaining to specific drivers based on driver IDs or names.
            - *By Customers:* Filter by specific customers or groups of customers.
            - *By Customer and Customer Locations:* Analyze data at the customer level and their respective locations.
            - *Volume Filters:* Use predefined delivery volume ranges (e.g., 1-100, 100-500 gallons).
            - *Status Filters:* Exclude or include records based on statuses like completed, pending, or canceled.
            - *Utilization Filters:* Narrow down records where utilization falls below a specified threshold.
            - *Null Handling:* Exclude or include null values as needed based on the analysis requirement.
        13. To enhance the SQL query:
            - Use subqueries or Common Table Expressions (CTEs) to preprocess data in the intermediary table and verify the accuracy of the join before aggregating data.
            - Ensure grouping is applied after all necessary joins to maintain the accuracy of the aggregated data.
        14. Get data from delivery report for comparative Analysis Rules
            - Total Stops = COUNT(DISTINCT delivery_id) per date
            - Total Gallons = SUM(Volume) per date
            - Time Periods:
                - Weekly: Current (last 7 days) vs Previous (7 days before that)
                - Monthly: Current calendar month vs Previous calendar month
                - Quarterly: Current calendar quarter vs Previous calendar quarter.
        15. Don't use LIMIT.
        16. 13. Avoid using aggregate functions (like AVG) directly with window functions (like LAG). Instead, calculate window function results in a subquery or CTE and apply the aggregate function in the outer query.
        """

    if contextual_memory:
        context += "\nPrevious queries and answers:\n"
        for i, memory in enumerate(contextual_memory):
            context += f"Query {i + 1}: {memory['question']}\nAnswer {i + 1}: {memory['answer']}\n"

    context += f"\nQuestion: {question}\nGenerate an accurate SQL query that:\n"
    context += """
    1. Handles relationships and joins between tables.
    2. Includes appropriate WHERE and HAVING clauses.
    3. Uses window functions and aggregates effectively.
    4. Returns only relevant data as per the question.
    5. Handles edge cases like missing data and duplicate entries.
    6. Don't use LIMIT.
"""
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
col1, col2 = st.columns([10, 1])  # Adjust width ratio (memory icon smaller)

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
                    print(query_result)
                else:
                    raise ValueError(f"Unexpected query result format: {query_result}")

                # Execute the SQL query
                with engine.connect() as connection:
                    result = connection.execute(text(sql_query))
                    rows = result.fetchall()
                    columns = result.keys()

                    # Convert rows to JSON
                    json_result = [dict(zip(columns, row)) for row in rows]
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