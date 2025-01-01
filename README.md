# KarmaOpsAI: Conversational Insights

KarmaOpsAI is a conversational agent built to provide detailed insights into organizational data using PostgreSQL, LangChain, OpenAI GPT, and Streamlit. The application allows users to query the database and visualize the results interactively. It generates SQL queries based on user inputs and displays results in various formats, including tables and visualizations.

## Features
- **Conversational Queries:** Users can ask questions in natural language, and the AI generates accurate SQL queries to fetch the data.
- **Data Visualization:** The results can be visualized using different types of graphs (e.g., bar charts, line charts, scatter plots, heatmaps).
- **Customizable Database Queries:** Users can easily add their database schema and table structures, and the AI will adapt accordingly.

## Technologies Used
- **PostgreSQL** for database management.
- **SQLAlchemy** for ORM and SQL query execution.
- **LangChain** for orchestrating the AI and SQL query generation.
- **OpenAI GPT-4** for natural language processing and generating SQL queries.
- **Streamlit** for the web interface.
- **Plotly** for data visualization.

## Requirements

### Python Libraries:
- **streamlit**
- **sqlalchemy**
- **langchain-community**
- **plotly**
- **pandas**
- **psycopg2**
- **os**

You can install these dependencies by running:

```bash
pip install streamlit sqlalchemy langchain-community plotly pandas psycopg2 os
```

### PostgreSQL Setup

- **Database:** PostgreSQL is used as the database to store and query data.
- **Docker:** The PostgreSQL database is managed using Docker.
- **DBeaver:** DBeaver is used to connect to the PostgreSQL instance, explore the schema, and modify data types of columns.

### Environment Variables
You’ll need to set the following environment variables to configure your database and OpenAI API key:

- `DB_USER`: The username to access your PostgreSQL instance.
- `DB_PASSWORD`: The password for the PostgreSQL user.
- `DB_HOST`: The host for the PostgreSQL instance (usually `localhost`).
- `DB_PORT`: The port for the PostgreSQL instance (default is `5432`).
- `DB_NAME`: The name of the PostgreSQL database.
- `OPENAI_API_KEY`: Your OpenAI API key.

You can set these variables in your terminal or create a `.env` file to load them automatically. Also set up your OpenAI API key in the OS as to not have it 
displayed while your showing the code.

Example `.env` file:

```
DB_USER=postgres
DB_PASSWORD=mysecretpassword
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
```

## Setting Up PostgreSQL with Docker

1. **Docker Setup:** Create a `docker-compose.yml` file to define the PostgreSQL container.

```yaml
version: "3.8"
services:
  db:
    image: postgres:latest
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: mysecretpassword
      POSTGRES_DB: postgres
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
volumes:
  pg_data:
```

2. **Start the Database:** Run the following command to start the PostgreSQL container.

```bash
docker-compose up -d
```

3. **Access the Database:** Connect to the PostgreSQL database via DBeaver or any other PostgreSQL client.

- Host: `localhost`
- Port: `5432`
- User: `postgres`
- Password: `mysecretpassword`
- Database: `postgres`

## Data Types in DBeaver

1. **Manual Data Type Modifications:** After setting up the database, you may need to change the data types of the columns in DBeaver manually. To do this:
   - Right-click the table in the Database Navigator and choose **Edit Table**.
   - Modify the data type of columns as necessary.
   - Save the changes.

2. **Schema Exploration:** Use DBeaver to explore the schemas and tables available in your PostgreSQL database. This information is used by the LangChain-based app to generate the correct SQL queries.

## Running the Application

1. **Set Up Environment Variables:** Ensure the environment variables are correctly set (as mentioned in the Environment Variables section).
   
2. **Run Streamlit App:**
   
```bash
streamlit run main.py
```

This will start a local server where you can interact with the application. Open the browser at `http://localhost:8501` to start querying the database.

## How It Works

1. **User Inputs Question:** The user inputs a question (e.g., "Who are the top sales performers?").
2. **SQL Query Generation:** The system generates an SQL query based on the user input, using LangChain and OpenAI GPT-4.
3. **SQL Execution:** The generated SQL query is executed on the PostgreSQL database using SQLAlchemy.
4. **Data Visualization:** The result is presented as a table and can be visualized using various types of graphs, such as bar charts, scatter plots, and heatmaps.

## Example Use Cases
- **Sales Analysis:** Query the top-performing sales employees, total sales by region, etc.
- **Asset Utilization:** View asset utilization across different locations.
- **Customer Insights:** Query customer data based on various filters like location, demographics, and purchase history.

## Troubleshooting

- **Connection Issues:** Ensure that your PostgreSQL database is running and the correct credentials are set in the environment variables.
- **Query Errors:** If you encounter an error in generating SQL queries, check the database schema and table information to ensure it matches what the app expects.

---
