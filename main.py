# Import required libraries and modules
import openai
import streamlit as st
from streamlit_chat import message
from agent import create_dataframe_agent, prompt_similarity_search, query_agent, Promotion, create_structured_output_agent, SalesGraph

import sys
from io import StringIO
import re
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from tabulate import tabulate
import json
import requests
from datetime import datetime
from gen_ai_hub.proxy.native.openai import chat

def date_to_epoch(date_string):
    """
    Convert a date string to epoch format.
    
    Args:
        date_string (str): A string representing the date in the format '%Y-%m-%d %H:%M:%S.%f' or '%Y-%m-%d'.
        
    Returns:
        str: A string representing the date in epoch format, e.g., "/Date(milliseconds)/".
    """
    date_format = "%Y-%m-%d %H:%M:%S.%f"
    try:
        date = datetime.strptime(date_string, date_format)
    except ValueError:
        date = datetime.strptime(date_string.split()[0], "%Y-%m-%d")
    milliseconds = int(date.timestamp() * 1000)
    return f"/Date({milliseconds})/"

def convert_time_to_ph0h0m0s(time_in_seconds):
    """
    Convert time in seconds to hours, minutes, and seconds format.
    
    Args:
        time_in_seconds (int): Time duration in seconds.
        
    Returns:
        str: A string representing the time duration in the format 'hh:mm:ss'.
    """
    hours = time_in_seconds // 3600
    minutes = (time_in_seconds % 3600) // 60
    seconds = time_in_seconds % 60
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"


# Patterns for identifying time durations
LAST_MONTH_PATTERN = 'last [0-9]+ month(s)*'
LAST_WEEK_PATTERN = 'last [0-9]+ week(s)*'
LAST_DAY_PATTERN = 'last [0-9]+ day(s)*'

# Load the dataset
df = pd.read_csv('dataset/promotion_dataset.csv')

# Create an agent from the CSV file.
df_agent = create_dataframe_agent(df)
dm_agent = create_structured_output_agent(Promotion)
dm_agent_for_graph = create_structured_output_agent(SalesGraph)

    
# Setting page title and header
st.set_page_config(page_title="Intelligent Promotion Planner", page_icon=":robot_face:")#,layout="wide")
st.markdown("<h1 style='text-align: center;'>Intelligent Promotion Planner</h1>", unsafe_allow_html=True)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    # List of initial system message
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a promotion planner expert."}
    ]

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Controls")
counter_placeholder = st.sidebar.empty()
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Reset everything if clear button is clicked
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": f"You are a promotion planner expert.Generate the output inline to the input. Don't provide generated python codes for the given input. Today is {datetime.now()}"}
    ]

def generate_response(prompt):
    """
    Generate a response based on user input.
    
    Args:
        prompt (str): User input prompt.
        
    Returns:
        str: Generated response.
    """
    st.session_state['messages'].append({"role": "user", "content": prompt})  # Add user input to messages
    kwargs = dict(model_name='gpt-35-turbo-16k', messages=st.session_state['messages'],temperature=0)  # Arguments for model
    with st.spinner("Generating..."):
        try:
            response = df_agent.run(f"Generate the output inline to query. Today {datetime.now()}. Query: {prompt}")  # Generate response
        except Exception as e:
            print(e)
            response = "I'm sorry, but I'm not sure what you're asking for. Could you please provide more information or clarify your request?"
    return response


# Container for chat history
response_container = st.container()
# Container for text box
container = st.container()

# Display initial system message
with response_container:
    message("How can I assist you today?", key="assistant")

# Function to call Pandas agent for user input
def call_pandas_agent(user_input):
    output = generate_response(user_input)
    if output:
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

def draw_sales_analysis_dashboard(df):
    df_st_sa = df.groupby('BusinessUnitType').agg({"SalesVolume" : "mean"}).reset_index().sort_values(by='SalesVolume', ascending=False)
    df_fa_sa = df.groupby('Category').agg({"SalesVolume" : "mean"}).reset_index().sort_values(by='SalesVolume', ascending=False)[:10]
    df_cl_sa = df.groupby('BusinessUnit').agg({"SalesVolume" : "mean"}).reset_index() 
            # chart color
    df_fa_sa['color'] = '#32cf76'
    df_fa_sa['color'][2:] = '#a5d993'
    df_cl_sa['color'] = '#c8d984'

            # chart
    fig = make_subplots(rows=2, cols=2, 
                    specs=[[{"type": "bar"}, {"type": "pie"}],
                           [{"colspan": 2}, None]],
                    column_widths=[0.6, 0.4], vertical_spacing=0.4, horizontal_spacing=0.1,
                    subplot_titles=("Top 10 Highest Product Sales", "Highest Sales in Stores", "Stores Vs Sales"))

    fig.add_trace(go.Bar(x=df_fa_sa['SalesVolume'], y=df_fa_sa['Category'], marker=dict(color= df_fa_sa['color']),
                    name='Category', orientation='h'), 
                    row=1, col=1)
    fig.add_trace(go.Pie(values=df_st_sa['SalesVolume'], labels=df_st_sa['BusinessUnitType'], name='Business Unit Type',
                     marker=dict(colors=['#68992f','#80b346','#9ac963','#c5e0a6','#e1edd3']), hole=0.7,
                     hoverinfo='label+percent+value', textinfo='label'), 
                    row=1, col=2)
    fig.add_trace(go.Bar(x=df_cl_sa['BusinessUnit'], y=df_cl_sa['SalesVolume'], 
                     marker=dict(color= df_cl_sa['color']), name='BusinessUnit'), 
                     row=2, col=1)

            # styling
    fig.update_yaxes(showgrid=False, ticksuffix=' ', categoryorder='total ascending', row=1, col=1)
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_xaxes(tickmode = 'array', tickvals=df_cl_sa.BusinessUnit, row=2, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    fig.update_layout(height=500, bargap=0.2,
                  margin=dict(b=0,r=20,l=20), xaxis=dict(tickmode='linear'),
                  title_text="Sales Analysis Dashboard",
                  template="plotly_white",
                  title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                  font=dict(color='#8a8d93'),
                  hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                  showlegend=False)
          
    return fig
    
def draw_plotly_chart(sales_graph):
    p_color = None
    if  len(sales_graph.color) != 0:
        p_color = sales_graph.color
    elif len(sales_graph.x_field_filter_by) != 0:
        p_color = sales_graph.x_field_filter_by

    
    filtered_df = get_filtered_df_by_date(sales_graph=sales_graph)
    if(filtered_df.empty):
        filtered_df = get_filtered_df_by_values(sales_graph=sales_graph)
    if(filtered_df.empty):
        filtered_df = get_unfiltered_df(sales_graph=sales_graph)
    if(not filtered_df.empty and len(sales_graph.product_name) != 0):  
        filtered_df = df[df["ProductName"].str.contains(sales_graph.product_name)]  
        sales_graph.graph_name = f"{sales_graph.graph_name} for {sales_graph.product_name}"
    print(filtered_df)
    if(filtered_df.empty):
        return None
    fdf = df if filtered_df.empty else filtered_df
    if sales_graph.graph_type.casefold()=="line":
        fig = px.scatter(fdf, x=sales_graph.x_field_name, color=p_color, y=sales_graph.y_field_name)
        
    elif sales_graph.graph_type.casefold()=="bar": 
        fig = px.bar(fdf, x=sales_graph.x_field_name, color=p_color, y=sales_graph.y_field_name)
        
    elif sales_graph.graph_type.casefold()=="histogram": 
        fig = px.histogram(fdf, x=sales_graph.x_field_name, color=p_color, y=sales_graph.y_field_name)
    
    elif sales_graph.graph_type.casefold()=="scatter": 
        fig = px.scatter(fdf, x=sales_graph.x_field_name, color=p_color, y=sales_graph.y_field_name)
         
    fig.update_layout(title = sales_graph.graph_name)
    return fig


def get_filtered_df_by_values(sales_graph):
    filtered_df = pd.DataFrame([])
    print("dataframe", sales_graph.x_field_name)
    if (sales_graph.x_field_name != 'Date' and len(sales_graph.x_field_selected_values) != 0):
        splited_select_values = sales_graph.x_field_selected_values.strip('][').split(',')
        splited_select_values = [x.strip(' ') for x in splited_select_values]
        filtered_df = df[df[sales_graph.x_field_name].isin(splited_select_values)]
    return filtered_df


def get_filtered_df_by_date(sales_graph):
    filtered_df = pd.DataFrame([])
    if (len(sales_graph.x_field_filter_by) != 0 
        and
          len(sales_graph.x_field_filter_from) != 0 and
          len(sales_graph.x_field_filter_to) != 0):
        print(sales_graph.x_field_filter_from)
        if re.search('weeks ago', sales_graph.x_field_filter_from, re.IGNORECASE):
            filtered_df = get_last_N_weeks(int(sales_graph.x_field_filter_from.split()[0]))
            
        elif re.search('days ago', sales_graph.x_field_filter_from, re.IGNORECASE):
            filtered_df = get_last_N_days(int(sales_graph.x_field_filter_from.split()[0]))

        elif re.search('months ago', sales_graph.x_field_filter_from, re.IGNORECASE):
            filtered_df = get_last_N_months(int(sales_graph.x_field_filter_from.split()[0]))
            
    elif (re.search('month', sales_graph.x_field_filter_by, re.IGNORECASE) or 
              (re.search('Date', sales_graph.x_field_name, re.IGNORECASE) and 
               re.search(LAST_MONTH_PATTERN, sales_graph.x_field_selected_values, re.IGNORECASE)) ):
            month_matched = re.search(LAST_MONTH_PATTERN, sales_graph.x_field_selected_values, re.IGNORECASE)
            if month_matched:
                filtered_df = get_last_N_months(int(month_matched.string.split()[1]))
                
    elif (re.search('week', sales_graph.x_field_filter_by, re.IGNORECASE) or 
            (re.search('Date', sales_graph.x_field_name, re.IGNORECASE) and 
            re.search(LAST_WEEK_PATTERN, sales_graph.x_field_selected_values, re.IGNORECASE)) ):
        month_matched = re.search(LAST_WEEK_PATTERN, sales_graph.x_field_selected_values, re.IGNORECASE)
        if month_matched:
            filtered_df = get_last_N_weeks(int(month_matched.string.split()[1]))
            
    elif (re.search('day', sales_graph.x_field_filter_by, re.IGNORECASE) or 
            (re.search('Date', sales_graph.x_field_name, re.IGNORECASE) and 
            re.search(LAST_DAY_PATTERN, sales_graph.x_field_selected_values, re.IGNORECASE)) ):
        month_matched = re.search(LAST_DAY_PATTERN, sales_graph.x_field_selected_values, re.IGNORECASE)
        if month_matched:
            filtered_df = get_last_N_days(int(month_matched.string.split()[1]))

    if not filtered_df.empty and sales_graph.x_field_filter_by!='Date':
        sales_graph.x_field_name = 'Date'
    return filtered_df

def get_unfiltered_df(sales_graph):
    if(sales_graph.x_field_selected_values=='' and sales_graph.x_field_filter_from=='' and sales_graph.x_field_filter_to==''):
        return df
    return pd.DataFrame([])

def get_last_N_days(n):
    return df[df.index.date > (datetime.now().date()- pd.Timedelta(days=n))]

def get_last_N_weeks(n):
   return df[df.index.date > (datetime.now().date()- pd.Timedelta(weeks=n))]

def get_last_N_months(n):
    return df[df.index.date > (datetime.now().date()- pd.DateOffset(months=n))]

def handle_submit_prompt_2(user_input, submit_button):
    if submit_button and user_input:
        if (re.search('sales analysis', user_input, re.IGNORECASE)) or re.search('dashbaord', user_input, re.IGNORECASE):
            fig = draw_sales_analysis_dashboard(df)
            st.session_state['past'].append(user_input)
            st.session_state.generated.append(fig)
            
        elif (re.search('trend', user_input, re.IGNORECASE) or 
                re.search('show', user_input, re.IGNORECASE) or
                re.search('plot', user_input, re.IGNORECASE)):
            PREFIX = f"Consider and match appropriate field name from the list.{list(df.columns)}"
            with st.spinner("Generating..."):
                try:
                    sales_graph = dm_agent_for_graph.invoke(f"{user_input}. {PREFIX}")
                    print(sales_graph)
                    fig = draw_plotly_chart(sales_graph=sales_graph)
                    st.session_state['past'].append(user_input)
                    st.session_state.generated.append("No results found for the given creteria. " if not fig else fig)
                except Exception as e:
                    print(e)
                    err_txt = "I'm sorry, but I'm not sure what you're asking for. Could you please provide more information or clarify your request?"
                    st.session_state['past'].append(user_input)
                    st.session_state.generated.append(err_txt)
    
        elif re.search('create', user_input, re.IGNORECASE) and re.search('promotion', user_input, re.IGNORECASE):
            with st.spinner("Processing the request..."):
                promo = dm_agent.invoke(f"Today is {datetime.now()}. {user_input}")
                print(promo)
                data = promo.__dict__

                selected_keys = ['promotionName', 'promoDescription', 'startDate', 'endDate', 'createdOn', 'logicalSystem', 'activeDate']
                selected_data = {key: data[key] for key in selected_keys if key in data}
                selected_data_bu = {'businessUnitID': data['businessUnitID']}
                selected_data['startDate'] = date_to_epoch(selected_data['startDate'])
                selected_data['activeDate'] = date_to_epoch(selected_data['activeDate'])
                selected_data['endDate'] = date_to_epoch(selected_data['endDate'])
                selected_data['createdOn'] = date_to_epoch(selected_data['createdOn'])

                default_data = {
                    "Promotion Planner Assistant - Chatbot": "<TENANT-ID>",
                    "expiryDate": "/Date(1701129599000)/",
                    "startTime":"PT0H0M0S",
                    "endTime":"PT23H59M59S",
                    "statusCode": "IA",
                    "origin": "10",
                    "version": 1
                }
                selected_data.update(default_data)

                url = '<APPLICATION-URL>'
                url_bu = '<APPLICATION-URL-BU>'
                CLIENT_ID = "<CLIENT-ID>"
                CLIENT_SECRET = "<CLIENT-SECRET>"
                TOKEN_URL = "<TOKEN-URL>"

                def generate_access_token(url, client_id, client_secret):
                    response = requests.post(
                        url,
                        data={"grant_type": "client_credentials"},
                        auth=(client_id, client_secret),
                    )
                    access_token = response.json()["access_token"]
                    return access_token

                gen_access_token = generate_access_token(TOKEN_URL, CLIENT_ID, CLIENT_SECRET)

                headers = {
                'Content-Type': 'application/json',
                'Accept-Language': 'en', 
                'accept': 'application/json',
                'Authorization': "Bearer {}".format(gen_access_token)
                }

                response = requests.post(url, headers=headers, data=json.dumps(selected_data))
                response_full = json.loads(response.content)

                promotion_id_bu = response_full["d"]["promotionID"]
                tenant_bu = response_full["d"]["tenant"]
                new_response = requests.get(url_bu, headers=headers)
                default_selected_data_bu = {
                    'tenant': tenant_bu,
                    'promotionID': promotion_id_bu,
                    'businessUnitType': 'RetailStore'
                }

                selected_data_bu.update(default_selected_data_bu)
                print(selected_data_bu)
                
                params = [('promotionID', ''), ('buID', '')]
                response_bu = requests.post(url_bu, headers=headers, data=json.dumps(selected_data_bu), params= params)
                print(response_bu.content)
            st.session_state['past'].append(user_input)
            st.session_state.generated.append(f"Promtion has been created successfully. Below are the details captured.\n\n {json.dumps(promo.__dict__, indent=2)}")

        else:
            call_pandas_agent(user_input)


with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    handle_submit_prompt_2(user_input, submit_button)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            generated_obj = st.session_state["generated"][i]
            from plotly.graph_objs import Figure
            try:
                if isinstance(generated_obj, Figure):
                    message(st.plotly_chart(generated_obj, use_container_width=False, sharing="streamlit", theme="streamlit"))
                else:
                    message(st.session_state["generated"][i], key=str(i))
            except Exception as e:
                    pass
