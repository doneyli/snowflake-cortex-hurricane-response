# need to add the following packages in SiS from the package picker: plotly, snowflake, snowflake-ml-python, snowflake-snowpark-python

import streamlit as st
from snowflake.core import Root  # requires snowflake>=0.8.0
from snowflake.cortex import Complete
from snowflake.snowpark.context import get_active_session

MODELS = [
    "mistral-large2",
    "llama3.1-70b",
    "llama3.1-8b",
]

def init_messages():
    """
    Initialize the session state for chat messages. If the session state indicates that the
    conversation should be cleared or if the "messages" key is not in the session state,
    initialize it as an empty list.
    """
    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = []


def init_service_metadata():
    """
    Initialize the session state for cortex search service metadata. Query the available
    cortex search services from the Snowflake session and store their names and search
    columns in the session state.
    """
    if "service_metadata" not in st.session_state:
        services = session.sql("SHOW CORTEX SEARCH SERVICES;").collect()
        service_metadata = []
        if services:
            for s in services:
                svc_name = s["name"]
                svc_search_col = session.sql(
                    f"DESC CORTEX SEARCH SERVICE {svc_name};"
                ).collect()[0]["search_column"]
                service_metadata.append(
                    {"name": svc_name, "search_column": svc_search_col}
                )

        st.session_state.service_metadata = service_metadata

def init_config_options():
    """
    Initialize the configuration options in the Streamlit sidebar. Allow the user to select
    a cortex search service, clear the conversation, toggle debug mode, and toggle the use of
    chat history. Also provide advanced options to select a model, the number of context chunks,
    and the number of chat messages to use in the chat history.
    """
    st.sidebar.selectbox(
        "Select cortex search service:",
        [s["name"] for s in st.session_state.service_metadata],
        key="selected_cortex_search_service",
    )

    st.sidebar.button("Clear conversation", key="clear_conversation")
    st.sidebar.toggle("Debug", key="debug", value=False)
    st.sidebar.toggle("Use chat history", key="use_chat_history", value=True)

    with st.sidebar.expander("Advanced options"):
        st.selectbox("Select model:", MODELS, key="model_name")
        st.number_input(
            "Select number of context chunks",
            value=10,
            key="num_retrieved_chunks",
            min_value=1,
            max_value=20,
        )
        st.number_input(
            "Select number of messages to use in chat history",
            value=5,
            key="num_chat_messages",
            min_value=1,
            max_value=10,
        )

    st.sidebar.expander("Session State").write(st.session_state)

def query_cortex_search_service(query, columns = [], filter={}):
    """
    Query the selected cortex search service with the given query and retrieve context documents.
    Display the retrieved context documents in the sidebar if debug mode is enabled. Return the
    context documents as a string.

    Args:
        query (str): The query to search the cortex search service with.

    Returns:
        str: The concatenated string of context documents.
    """
    db, schema = session.get_current_database(), session.get_current_schema()

    cortex_search_service = (
        root.databases[db]
        .schemas[schema]
        .cortex_search_services[st.session_state.selected_cortex_search_service]
    )

    context_documents = cortex_search_service.search(
       #query, columns=columns, filter=filter, limit=st.session_state.num_retrieved_chunks
        query, columns=columns, limit=st.session_state.num_retrieved_chunks
    )
    results = context_documents.results

    service_metadata = st.session_state.service_metadata
    search_col = [s["search_column"] for s in service_metadata
                    if s["name"] == st.session_state.selected_cortex_search_service][0].lower()

    context_str = ""
    for i, r in enumerate(results):
        context_str += f"Context document {i+1}: {r[search_col]} \n" + "\n"

    if st.session_state.debug:
        st.sidebar.text_area("Context documents", context_str, height=500)

    return context_str, results


def get_chat_history():
    """
    Retrieve the chat history from the session state limited to the number of messages specified
    by the user in the sidebar options.

    Returns:
        list: The list of chat messages from the session state.
    """
    start_index = max(
        0, len(st.session_state.messages) - st.session_state.num_chat_messages
    )
    return st.session_state.messages[start_index : len(st.session_state.messages) - 1]


def complete(model, prompt):
    """
    Generate a completion for the given prompt using the specified model.

    Args:
        model (str): The name of the model to use for completion.
        prompt (str): The prompt to generate a completion for.

    Returns:
        str: The generated completion.
    """
    return Complete(model, prompt).replace("$", "\$")


def make_chat_history_summary(chat_history, question):
    """
    Generate a summary of the chat history combined with the current question to extend the query
    context. Use the language model to generate this summary.

    Args:
        chat_history (str): The chat history to include in the summary.
        question (str): The current user question to extend with the chat history.

    Returns:
        str: The generated summary of the chat history and question.
    """
    prompt = f"""
        [INST]
        Based on the chat history below and the question, generate a query that extend the question
        with the chat history provided. The query should be in natural language.
        Answer with only the query. Do not add any explanation.

        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        [/INST]
    """

    summary = complete(st.session_state.model_name, prompt)

    if st.session_state.debug:
        st.sidebar.text_area(
            "Chat history summary", summary.replace("$", "\$"), height=150
        )

    return summary


def create_prompt(user_question):
    """
    Create a prompt for the language model by combining the user question with context retrieved
    from the cortex search service and chat history (if enabled). Format the prompt according to
    the expected input format of the model.

    Args:
        user_question (str): The user's question to generate a prompt for.

    Returns:
        str: The generated prompt for the language model.
    """
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()
        if chat_history != []:
            question_summary = make_chat_history_summary(chat_history, user_question)
            prompt_context, results = query_cortex_search_service(
                question_summary,
                columns=["incident_text"],
                #filter={"@and": [{"@eq": {"language": "English"}}]},
            )
        else:
            prompt_context, results = query_cortex_search_service(
                user_question,
                columns=["incident_text"],
                #filter={"@and": [{"@eq": {"language": "English"}}]},
            )
    else:
        prompt_context, results = query_cortex_search_service(
            user_question,
            columns=["incident_text"],
            #filter={"@and": [{"@eq": {"language": "English"}}]},
        )
        chat_history = ""

    prompt = f"""
            [INST]
            You are a helpful AI chat assistant with RAG capabilities. When a user asks you a question,
            you will also be given context provided between <context> and </context> tags. Use that context
            with the user's chat history provided in the between <chat_history> and </chat_history> tags
            to provide a summary that addresses the user's question. Ensure the answer is coherent, concise,
            and directly relevant to the user's question.

            If the user asks a generic question which cannot be answered with the given context or chat_history,
            just say "I don't know the answer to that question.

            Don't saying things like "according to the provided context".

            <chat_history>
            {chat_history}
            </chat_history>
            <context>
            {prompt_context}
            </context>
            <question>
            {user_question}
            </question>
            [/INST]
            Answer:
            """
    return prompt, results

def display_outages_categories_results():
    """Displays the OUTAGES_GENAI_CATEGORIES_COUNT and OUTAGES_GENAI_PARTS_COUNT tables with chart options."""
    st.title("Outages Analytics")

    # Query the OUTAGES_GENAI_CATEGORIES_COUNT table
    categories_results = session.sql("SELECT CATEGORY, SUM(OCCURRENCE_COUNT) AS COUNT FROM OUTAGES_GENAI_CATEGORIES_COUNT GROUP BY CATEGORY ORDER BY SUM(OCCURRENCE_COUNT) DESC").collect()

    # Convert results to a Pandas DataFrame for charting
    import pandas as pd
    categories_data = pd.DataFrame(categories_results)

    if categories_data.empty:
        st.warning("No data found in OUTAGES_GENAI_CATEGORIES_COUNT table.")
        return

    # Display categories table
    st.write("Category Data")
    st.dataframe(categories_data)

    # Chart options for OUTAGES_GENAI_CATEGORIES_COUNT
    st.write("Visualize the Categories Data")
    chart_type_categories = st.radio(
        "Select a chart type for Categories Data:",
        ["Bar Chart", "Pie Chart", "Line Chart", "Table Only"],
        index=0
    )

    if chart_type_categories == "Bar Chart":
        st.bar_chart(categories_data.set_index("CATEGORY"))
    elif chart_type_categories == "Pie Chart":
        import plotly.express as px
        fig = px.pie(categories_data, names="CATEGORY", values="COUNT", title="Category Distribution")
        st.plotly_chart(fig)
    elif chart_type_categories == "Line Chart":
        st.line_chart(categories_data.set_index("CATEGORY"))
    else:
        st.write("Data Table Only")
        st.dataframe(categories_data)

    # Spacer for clarity
    st.write("---")

    # Query the OUTAGES_GENAI_PARTS_COUNT table
    parts_results = session.sql("SELECT PART, PART_COUNT FROM OUTAGES_GENAI_PARTS_COUNT").collect()

    # Convert results to a Pandas DataFrame
    parts_data = pd.DataFrame(parts_results)

    if parts_data.empty:
        st.warning("No data found in OUTAGES_GENAI_PARTS_COUNT table.")
        return

    # Display parts table
    st.write("Parts Data")
    st.dataframe(parts_data)

    # Chart options for OUTAGES_GENAI_PARTS_COUNT
    st.write("Visualize the Parts Data")
    chart_type_parts = st.radio(
        "Select a chart type for Parts Data:",
        ["Bar Chart", "Pie Chart", "Line Chart", "Table Only"],
        index=0
    )

    if chart_type_parts == "Bar Chart":
        st.bar_chart(parts_data.set_index("PART"))
    elif chart_type_parts == "Pie Chart":
        import plotly.express as px
        fig = px.pie(parts_data, names="PART", values="PART_COUNT", title="Parts Distribution")
        st.plotly_chart(fig)
    elif chart_type_parts == "Line Chart":
        st.line_chart(parts_data.set_index("PART"))
    else:
        st.write("Data Table Only")
        st.dataframe(parts_data)


def display_category_locations_map():
    """Displays a map with the locations of categories based on their LATITUDE and LONGITUDE."""
    st.title("Outages Category Map")

    # Query the OUTAGES_GENAI_CATEGORIES_COUNT table for location data
    location_results = session.sql("""
        SELECT LATITUDE, LONGITUDE, CATEGORY, OCCURRENCE_COUNT
        FROM CLASSIFICATION_DB.HURRICANE_SCHEMA.OUTAGES_GENAI_CATEGORIES_COUNT
    """).collect()

    # Convert results to a Pandas DataFrame for mapping
    import pandas as pd
    location_data = pd.DataFrame(location_results)

    if location_data.empty:
        st.warning("No location data found in OUTAGES_GENAI_CATEGORIES_COUNT table.")
        return

    # Ensure columns match expectations
    location_data.columns = ["LATITUDE", "LONGITUDE", "CATEGORY", "OCCURRENCE_COUNT"]

    # Display the map
    st.write("Map of Outage Categories")
    st.map(location_data[["LATITUDE", "LONGITUDE"]])

    # Add a data table for more details
    st.write("Category Locations Data")
    st.dataframe(location_data)

    # Allow filtering by category
    selected_category = st.selectbox(
        "Filter by Category",
        options=location_data["CATEGORY"].unique(),
        index=0
    )
    filtered_data = location_data[location_data["CATEGORY"] == selected_category]

    # Display filtered map and table
    st.write(f"Map for Category: {selected_category}")
    st.map(filtered_data[["LATITUDE", "LONGITUDE"]])
    st.write("Filtered Data Table")
    st.dataframe(filtered_data)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Outages Chat", "Outages Analytics", "Outages Map"])

    if page == "Outages Chat":
        st.title(":thunder_cloud_and_rain: Hurricane Response with Snowflake Cortex")
        init_service_metadata()
        init_config_options()
        init_messages()

        icons = {"assistant": "❄️", "user": "👤"}

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=icons[message["role"]]):
                st.markdown(message["content"])

        disable_chat = (
            "service_metadata" not in st.session_state
            or len(st.session_state.service_metadata) == 0
        )
        if question := st.chat_input("Ask a question...", disabled=disable_chat):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            # Display user message in chat message container
            with st.chat_message("user", avatar=icons["user"]):
                st.markdown(question.replace("$", "\$"))

            # Display assistant response in chat message container
            with st.chat_message("assistant", avatar=icons["assistant"]):
                message_placeholder = st.empty()
                question = question.replace("'", "")
                prompt, results = create_prompt(question)
                with st.spinner("Thinking..."):
                    generated_response = complete(
                        st.session_state.model_name, prompt
                    )

                # build references table for citation
                markdown_table = "###### References \n\n| PDF Title | URL |\n|-------|-----|\n"
                for ref in results:
                    markdown_table += f"| {ref} | [Link]({ref}) |\n"
                message_placeholder.markdown(generated_response + "\n\n")

            
            st.session_state.messages.append(
                {"role": "assistant", "content": generated_response}
            )

    elif page == "Outages Analytics":
        display_outages_categories_results()

    elif page == "Outages Map":
        display_category_locations_map()


if __name__ == "__main__":
    session = get_active_session()
    root = Root(session)
    main()
