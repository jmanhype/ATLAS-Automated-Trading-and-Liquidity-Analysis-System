# ATLAS: Automated Trading and Liquidity Analysis System

## Overview

ATLAS (Automated Trading and Liquidity Analysis System) is a sophisticated multi-agent system designed for automated market making and trading. It leverages various specialized agents to handle different aspects of market analysis, order execution, risk management, and performance monitoring.

## Components

1. **MarketDataAgent**: Retrieves real-time market data from Yahoo Finance.
2. **LiquidityAnalysisAgent**: Performs in-depth liquidity analysis and calculates market spreads.
3. **OrderPlacementAgent**: Executes buy and sell orders based on current market conditions and analysis.
4. **RiskManagementAgent**: Continuously assesses and manages trading risks.
5. **PerformanceMonitoringAgent**: Tracks and calculates key performance metrics for the trading system.
6. **VisualizationAgent**: Generates comprehensive charts and visualizations of market data and system performance.

## Prerequisites

- Python 3.7+
- PostgreSQL database

## Installation

1. Clone the ATLAS repository:
   ```
   git clone https://github.com/yourusername/ATLAS.git
   cd ATLAS-Automated-Trading-and-Liquidity-Analysis-System
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   DB_USER=your_database_user
   DB_PASSWORD=your_database_password
   DB_HOST=your_database_host
   DB_PORT=your_database_port
   DB_NAME=your_database_name
   ```

## Usage

To start the ATLAS system, run the main script:

```
python atlas_main.py
```

The system will initiate multiple cycles of data fetching, market analysis, order placement, risk assessment, and performance monitoring. Visualization artifacts will be generated in HTML format for easy interpretation of results.

## Project Structure

- `atlas_main.py`: Main script that orchestrates the ATLAS system.
- `agents/`: Directory containing individual agent implementations.
- `data_repository.py`: Manages data storage and retrieval using PostgreSQL.
- `versions/`: Directory for storing generated code versions.
- `requirements.txt`: List of Python dependencies.

## Customization

- Modify agent behaviors by editing their respective classes in the `agents/` directory.
- Adjust risk parameters, order sizes, and other constants in the agent implementations to fit your trading strategy.
- Extend the system by adding new agent types or enhancing existing ones to incorporate additional analysis or trading strategies.

## Contributing

Contributions to enhance ATLAS are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/new-analysis-method`)
3. Commit your changes (`git commit -m 'Add new market analysis method'`)
4. Push to the branch (`git push origin feature/new-analysis-method`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Disclaimer

ATLAS is designed for educational and research purposes in the field of automated trading and market analysis. It is not intended for use in live trading without proper risk management protocols and thorough testing. Always consult with a financial advisor and comply with relevant regulations before making investment decisions or deploying automated trading systems.
