import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from transitions import Machine
import asyncio
import os
from dspy.functional import TypedPredictor
from typing import TypedDict
from dotenv import load_dotenv
import dspy
from dspy.signatures import InputField, OutputField
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pickle
from sqlalchemy.types import LargeBinary
from datetime import datetime, timedelta

load_dotenv()

# Create a language model using the OpenAI API
llm = dspy.OpenAI(
    model='gpt-4o',
    api_key=os.environ['OPENAI_API_KEY'],
    max_tokens=2000
)
# Configure DSPy to use this language model
dspy.configure(lm=llm)

class DecisionSignature(dspy.Signature):
    input_text: str = InputField()
    decision: bool = OutputField()
    explanation: str = OutputField()

class DataRepository:
    def __init__(self, db_config, schema_name):
        self.engine = create_engine(f'postgresql+psycopg2://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{db_config["database"]}')
        self.schema_name = schema_name
        self._ensure_schema()

    def _ensure_schema(self):
        try:
            with self.engine.begin() as connection:
                schema_check_query = f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{self.schema_name}'"
                result = connection.execute(text(schema_check_query)).fetchone()
                if result is None:
                    print(f"Schema '{self.schema_name}' does not exist. Creating schema.")
                    connection.execute(text(f"CREATE SCHEMA {self.schema_name}"))
                else:
                    print(f"Schema '{self.schema_name}' already exists.")
        except SQLAlchemyError as e:
            print(f"Error during schema creation: {e}")
            raise

    def store_data(self, data, table_name):
        data.to_sql(table_name, self.engine, schema=self.schema_name, if_exists='replace', index=False)

    def retrieve_data(self, table_name):
        return pd.read_sql_table(table_name, self.engine, schema=self.schema_name)

    def store_model(self, model, table_name):
        serialized_model = pickle.dumps(model)
        df = pd.DataFrame({'model': [serialized_model]})
        df.to_sql(table_name, self.engine, schema=self.schema_name, if_exists='replace', index=False, dtype={'model': LargeBinary})

    def retrieve_model(self, table_name):
        df = pd.read_sql_table(table_name, self.engine, schema=self.schema_name)
        serialized_model = df['model'][0]
        return pickle.loads(serialized_model)

# Initialize the DataRepository
db_config = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME")
}
data_repo = DataRepository(db_config, "market_maker")

class Agent(Machine):
    def __init__(self, llm, name, role=None, skills=None):
        self.llm = llm
        self.name = name
        self.role = role
        self.skills = skills or []
        self.memory = []
        states = ['start', 'thought', 'asked_question', 'acted', 'observed', 'reviewed', 'concluded']
        Machine.__init__(self, states=states, initial='start')
        self.add_transition('think', 'start', 'thought')
        self.add_transition('ask_question', 'thought', 'asked_question')
        self.add_transition('act', ['thought', 'asked_question'], 'acted')
        self.add_transition('observe', 'acted', 'observed')
        self.add_transition('review', 'observed', 'reviewed')
        self.add_transition('decide', 'reviewed', ['start', 'concluded'])

    async def think(self):
        print(f"{self.name} is thinking...")
        prompt = f"{self.name}, considering your role: {self.role}, and skills: {self.skills}, think step by step about how to generate code that fulfills the task objective."
        response = self.llm(prompt).pop()
        self.memory.append(response)
        self.state = 'thought'
        print(response)

    async def ask_question(self, question):
        print(f"{self.name} is asking a question...")
        prompt = f"{self.name}, you have the following question: {question}. Provide an answer based on your role: {self.role} and skills: {self.skills}."
        response = self.llm(prompt).pop()
        self.memory.append(response)
        self.state = 'asked_question'
        print(response)

    async def act(self, data):
        print("Data shape:", data.shape)
        print("Data columns:", data.columns)
        print("First few rows of data:")
        print(data.head())

        if data.empty:
            print("No data available. Skipping order placement.")
            return data

        if 'Close' not in data.columns:
            print("'Close' column not found in data. Skipping order placement.")
            return data

        last_price = data['Close'].iloc[-1] if not data['Close'].empty else None
        if last_price is None:
            print("No closing price available. Skipping order placement.")
            return data

        print(f"{self.name} is placing orders...")
        spread = data['Spread'].iloc[-1] if 'Spread' in data.columns else 0.01  # Default spread if not available
        bid_price = last_price - spread / 2
        ask_price = last_price + spread / 2
        order_size = 100  # Simplified order size
        
        buy_order = {'price': bid_price, 'size': order_size, 'side': 'buy'}
        sell_order = {'price': ask_price, 'size': order_size, 'side': 'sell'}
        
        orders = pd.DataFrame([buy_order, sell_order])
        self.memory.append(orders)
        data_repo.store_data(orders, 'placed_orders')
        print(f"Order Placement Agent placed orders and stored them.")
        self.state = 'acted'
        return orders
        # This method should be overridden by subclasses

    async def observe(self):
        print(f"{self.name} is observing...")
        self.state = 'observed'

    async def review(self, feedback):
        print(f"{self.name} is reviewing feedback...")
        self.memory.append(feedback)
        self.state = 'reviewed'

    async def decide(self):
        print(f"{self.name} is deciding...")
        str_memory = ' '.join(str(item) for item in self.memory)
        prompt = f"Based on your observations and feedback:\n\n{str_memory}\n\nMake a decision on the final code. Take into account whether the code fulfills the task objective."
        
        # Initialize TypedPredictor without the model parameter
        decision_maker = TypedPredictor(DecisionSignature)
        
        try:
            # Use the language model directly when calling the predictor
            response = decision_maker(input_text=prompt, lm=self.llm)
            if response.decision:
                self.state = 'concluded'
                final_code = self.llm(f"What is the final code that fulfills this task objective, given this information: {str_memory}").pop()
                print(f"{self.name}'s final code is: {final_code}")

                version_dir = f"versions/{self.name}_version"
                if not os.path.exists(version_dir):
                    os.makedirs(version_dir)
                with open(f"{version_dir}/final_code.txt", "w", encoding='utf-8') as file:
                    file.write(final_code)

                return final_code
            else:
                print(f"Decision not reached by {self.name}")
        except ValueError as e:
            print(f"Error in decision making: {e}")
        self.state = 'start'
        self.memory.append(f"Decision not reached by {self.name} because the analysis could not be completed.")
        return None

    async def execute(self, data=None):
        while self.state != 'concluded':
            if self.state == 'start':
                await self.think()
            elif self.state == 'thought':
                question = self.llm(f"{self.name}, based on your current thoughts: {' '.join(str(item) for item in self.memory)}, what question do you have that would help you generate better code?").pop()
                if question:
                    await self.ask_question(question)
                else:
                    data = await self.act(data)
            elif self.state == 'asked_question':
                data = await self.act(data)
            elif self.state == 'acted':
                await self.observe()
            elif self.state == 'observed':
                await self.review("No feedback available.")
            elif self.state == 'reviewed':
                return await self.decide()

class MarketDataAgent(Agent):
    def __init__(self, llm, name):
        super().__init__(llm, name, role="Market Data Agent", skills=["data fetching", "data storage"])

    async def act(self, data=None):
        print(f"{self.name} is fetching market data...")
        
        # Use a more recent date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Fetch data for the last 7 days
        
        try:
            data = yf.download("AAPL", start=start_date, end=end_date, interval="1m")
            print(f"Data shape: {data.shape}")
            print(f"Data columns: {data.columns}")
            print(f"Data index: {data.index}")
            
            if data.empty:
                print("No data fetched from Yahoo Finance. The market might be closed for the selected period.")
                return pd.DataFrame()
            
            self.memory.append(data)
            data_repo.store_data(data, 'market_data')
            print(f"Market Data Agent fetched and stored market data. Shape: {data.shape}")
            self.state = 'acted'
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

class LiquidityAnalysisAgent(Agent):
    def __init__(self, llm, name):
        super().__init__(llm, name, role="Liquidity Analysis Agent", skills=["liquidity analysis", "spread analysis"])

    async def act(self, data=None):
        print(f"{self.name} is analyzing liquidity...")
        data = data_repo.retrieve_data('market_data')
        if isinstance(data, pd.DataFrame):
            if 'Ask' in data.columns and 'Bid' in data.columns:
                data['Spread'] = data['Ask'] - data['Bid']
            else:
                print("Warning: 'Ask' or 'Bid' columns not found. Using 'High' and 'Low' as a proxy.")
                data['Spread'] = data['High'] - data['Low']
            
            data['Liquidity'] = data['Volume'] / data['Spread']
            self.memory.append(data)
            data_repo.store_data(data, 'liquidity_data')
            print(f"Liquidity Analysis Agent analyzed liquidity and stored results.")
            self.state = 'acted'
            return data
        else:
            print(f"Liquidity Analysis Agent received invalid data type: {type(data)}")

class OrderPlacementAgent(Agent):
    def __init__(self, llm, name):
        super().__init__(llm, name, role="Order Placement Agent", skills=["order placement", "spread capture"])

    async def act(self, data=None):
        print(f"{self.name} is placing orders...")
        data = data_repo.retrieve_data('liquidity_data')
        if isinstance(data, pd.DataFrame):
            last_price = data['Close'].iloc[-1]
            spread = data['Spread'].iloc[-1]
            bid_price = last_price - spread / 2
            ask_price = last_price + spread / 2
            order_size = 100  # Simplified order size
            
            buy_order = {'price': bid_price, 'size': order_size, 'side': 'buy'}
            sell_order = {'price': ask_price, 'size': order_size, 'side': 'sell'}
            
            orders = pd.DataFrame([buy_order, sell_order])
            self.memory.append(orders)
            data_repo.store_data(orders, 'placed_orders')
            print(f"Order Placement Agent placed orders and stored them.")
            self.state = 'acted'
            return orders
        else:
            print(f"Order Placement Agent received invalid data type: {type(data)}")

class RiskManagementAgent(Agent):
    def __init__(self, llm, name):
        super().__init__(llm, name, role="Risk Management Agent", skills=["risk assessment", "position management"])

    async def act(self, data=None):
        print(f"{self.name} is managing risk...")
        orders = data_repo.retrieve_data('placed_orders')
        market_data = data_repo.retrieve_data('market_data')
        if isinstance(orders, pd.DataFrame) and isinstance(market_data, pd.DataFrame):
            last_price = market_data['Close'].iloc[-1]
            net_position = orders[orders['side'] == 'buy']['size'].sum() - orders[orders['side'] == 'sell']['size'].sum()
            exposure = net_position * last_price
            
            risk_assessment = {
                'net_position': net_position,
                'exposure': exposure,
                'max_allowed_exposure': 10000  # Simplified risk limit
            }
            
            self.memory.append(risk_assessment)
            data_repo.store_data(pd.DataFrame([risk_assessment]), 'risk_assessment')
            print(f"Risk Management Agent assessed risk and stored results.")
            self.state = 'acted'
            return risk_assessment
        else:
            print(f"Risk Management Agent received invalid data type.")

class PerformanceMonitoringAgent(Agent):
    def __init__(self, llm, name):
        super().__init__(llm, name, role="Performance Monitoring Agent", skills=["performance tracking", "profit calculation"])

    async def act(self, data=None):
        print(f"{self.name} is monitoring performance...")
        orders = data_repo.retrieve_data('placed_orders')
        market_data = data_repo.retrieve_data('market_data')
        if isinstance(orders, pd.DataFrame) and isinstance(market_data, pd.DataFrame):
            last_price = market_data['Close'].iloc[-1]
            buy_orders = orders[orders['side'] == 'buy']
            sell_orders = orders[orders['side'] == 'sell']
            
            buy_value = (buy_orders['price'] * buy_orders['size']).sum()
            sell_value = (sell_orders['price'] * sell_orders['size']).sum()
            current_value = (buy_orders['size'].sum() - sell_orders['size'].sum()) * last_price
            
            profit_loss = sell_value - buy_value + current_value
            
            performance = {
                'profit_loss': profit_loss,
                'num_trades': len(orders),
                'average_spread': orders['price'].diff().abs().mean()
            }
            
            self.memory.append(performance)
            data_repo.store_data(pd.DataFrame([performance]), 'performance_data')
            print(f"Performance Monitoring Agent calculated performance and stored results.")
            self.state = 'acted'
            return performance
        else:
            print(f"Performance Monitoring Agent received invalid data type.")

class VisualizationAgent(Agent):
    def __init__(self, llm, name):
        super().__init__(llm, name, role="Visualization Agent", skills=["data visualization", "chart creation"])

    async def act(self, data=None):
        print(f"{self.name} is creating visualizations...")
        market_data = data_repo.retrieve_data('market_data')
        liquidity_data = data_repo.retrieve_data('liquidity_data')
        performance_data = data_repo.retrieve_data('performance_data')

        if isinstance(market_data, pd.DataFrame) and isinstance(liquidity_data, pd.DataFrame) and isinstance(performance_data, pd.DataFrame):
            # Create price chart
            fig1 = go.Figure(data=[go.Candlestick(x=market_data.index,
                                                  open=market_data['Open'],
                                                  high=market_data['High'],
                                                  low=market_data['Low'],
                                                  close=market_data['Close'])])
            fig1.update_layout(title='Price Chart', xaxis_title='Date', yaxis_title='Price')
            fig1.write_html("price_chart.html")

            # Create liquidity chart
            fig2 = go.Figure(data=[go.Scatter(x=liquidity_data.index, y=liquidity_data['Liquidity'], mode='lines')])
            fig2.update_layout(title='Liquidity Chart', xaxis_title='Date', yaxis_title='Liquidity')
            fig2.write_html("liquidity_chart.html")

            # Create performance chart
            fig3 = go.Figure(data=[go.Bar(x=['Profit/Loss', 'Number of Trades', 'Average Spread'],
                                          y=[performance_data['profit_loss'].iloc[-1],
                                             performance_data['num_trades'].iloc[-1],
                                             performance_data['average_spread'].iloc[-1]])])
            fig3.update_layout(title='Performance Metrics', xaxis_title='Metric', yaxis_title='Value')
            fig3.write_html("performance_chart.html")

            self.memory.append(["price_chart.html", "liquidity_chart.html", "performance_chart.html"])
            data_repo.store_data(pd.DataFrame({'charts': ["price_chart.html", "liquidity_chart.html", "performance_chart.html"]}), 'visualization')
            print(f"Visualization Agent created charts and stored them.")
            self.state = 'acted'
            return ["price_chart.html", "liquidity_chart.html", "performance_chart.html"]
        else:
            print(f"Visualization Agent received invalid data type.")

async def run_market_maker_system(llm):
    agents = [
        MarketDataAgent(llm, "Market Data Agent"),
        LiquidityAnalysisAgent(llm, "Liquidity Analysis Agent"),
        OrderPlacementAgent(llm, "Order Placement Agent"),
        RiskManagementAgent(llm, "Risk Management Agent"),
        PerformanceMonitoringAgent(llm, "Performance Monitoring Agent"),
        VisualizationAgent(llm, "Visualization Agent")
    ]

    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        for agent in agents:
            await agent.execute()
        
        retry_count += 1
        if retry_count < max_retries:
            print(f"Attempt {retry_count} completed. Waiting before next attempt...")
            await asyncio.sleep(60)  # Wait for 60 seconds before the next attempt

    print("Max retries reached. Exiting.")

# Usage
# llm = YourLanguageModel()  # Initialize your language model
# asyncio.run(run_market_maker_system(llm))
if __name__ == "__main__":
    asyncio.run(run_market_maker_system(llm))
