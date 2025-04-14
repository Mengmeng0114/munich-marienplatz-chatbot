# Munich Marienplatz Assistant

An intelligent chatbot assistant that provides information about parking facilities and sushi restaurants around Marienplatz in Munich.

## Overview

This project implements a conversational AI system that helps users find and get information about parking facilities and sushi restaurants in the Marienplatz area. The system uses natural language processing to understand user queries and provides relevant, contextual responses.

## Features

- Natural language understanding for parking and restaurant queries
- Semantic search capabilities for finding relevant locations
- Intelligent ranking of results based on query context
- Conversational memory for maintaining context
- Support for various query types (recommendations, details, comparisons, availability)

## Technical Architecture

The system is built with a modular architecture:

- **Data Processing**: Handles JSON data for locations
- **Intent Recognition**: Uses OpenAI GPT to understand user queries
- **Vector Embedding**: Implements semantic search using OpenAI embeddings
- **Information Retrieval**: Combines vector and keyword-based search
- **Response Generation**: Creates natural, contextual responses
- **Conversation Management**: Maintains conversation history and context

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

## Usage

Run the main script:
```bash
python chatbot_test.py
```

The system will respond to queries about:
- Parking facilities (availability, prices, restrictions)
- Sushi restaurants (ratings, prices, cuisine types)

## Data Privacy Notice

**Important**: The location data used in this project contains sensitive information and is subject to data protection regulations. Therefore, the actual data files are not included in this repository. Users need to obtain appropriate data access permissions before using this system.

## Dependencies

- OpenAI API (GPT-3.5-turbo, text-embedding-ada-002)
- Python 3.8+
- pandas
- numpy
- scipy
- tenacity

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 