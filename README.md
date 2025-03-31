# 🧩 StructSense

Welcome to `structsense`!

`structsense` is a powerful multi-agent system designed to extract structured information from unstructured data. By orchestrating intelligent agents, it helps you make sense of complex information — hence the name *structsense*.

Whether you're working with scientific texts, documents, or messy data, `structsense` enables you to transform it into meaningful, structured insights.

**Caution:**: this package is still under development and may change rapidly over the next few weeks.

---
## 🏗️ Architecture
The below is the architecture of the `StructSense`.

![](structsense_arch.png)
---
## 🚀 Features

- 🔍 Multi-agent architecture for modular processing
- 📑 Extraction of (structured) information from text--based on configuration
- 🤝 Collaboration between agents
- ⚙️ Easy use
- 🧠 Designed as general purpose domain agnostic framework

---

## 🧠 Example Use Cases
- Entity and relation extraction from text
- Knowledge graph construction
---
## 📄 Requirements
### GROBID Service
You need GROBID service running if you plan to use PDF files as input. For the raw input text, you do not use GROBID.

```shell
docker pull lfoppiano/grobid:0.8.0
docker run --init -p 8070:8070 -e JAVA_OPTS="-XX:+UseZGC" lfoppiano/grobid:0.8.0
```
JAVA_OPTS="-XX:+UseZGC" helps to resolve the following error in MAC OS.

---

## 📄 Configuration
`structsense` supports flexible customization through both environment variables and a YAML configuration file.

The YAML config can be passed as a parameter (e.g., `--agentconfig config/ner_agent.yaml`), allowing you to define models, agents, and behaviors specific to your use case.

### 🔧 Environment Variables

You need to set the following environment variables (e.g., in a `.env` file). WEAVIATE is a vector database that we use to store the knolwledge, which in our case is the ontology/schemas.

#### 🧠 Core Keys

| Variable              | Description                                  | Default          |
|-----------------------|----------------------------------------------|------------------|
 | `ENABLE_KG_SOURCE`    | Enable access to knowledge source, i.e., vector database.| `false`|
| `WEAVIATE_API_KEY`    | **Required.** API key for Weaviate access    | —                |

#### 🌐 [Weaviate](https://weaviate.io/) Configuration

| Variable                   | Description                                  | Default   |
|---------------------------|----------------------------------------------|-----------|
| `WEAVIATE_HTTP_HOST`      | HTTP host for Weaviate                       | `localhost` |
| `WEAVIATE_HTTP_PORT`      | HTTP port for Weaviate                       | `8080`    |
| `WEAVIATE_HTTP_SECURE`    | Use HTTPS for HTTP connection (`true/false`) | `false`   |
| `WEAVIATE_GRPC_HOST`      | gRPC host for Weaviate                       | `localhost` |
| `WEAVIATE_GRPC_PORT`      | gRPC port for Weaviate                       | `50051`   |
| `WEAVIATE_GRPC_SECURE`    | Use secure gRPC (`true/false`)              | `false`   |

#### 🧪 Weaviate Timeouts 

| Variable                   | Description                                  | Default   |
|---------------------------|----------------------------------------------|-----------|
| `WEAVIATE_TIMEOUT_INIT`   | Timeout for initialization (in seconds)     | `30`      |
| `WEAVIATE_TIMEOUT_QUERY`  | Timeout for query operations (in seconds)   | `60`      |
| `WEAVIATE_TIMEOUT_INSERT` | Timeout for data insertions (in seconds)    | `120`     |

#### 🤖 Ollama Configuration for WEAVIATE

| Variable              | Description                                   | Default                                 |
|-----------------------|-----------------------------------------------|-----------------------------------------|
| `OLLAMA_API_ENDPOINT` | API endpoint for Ollama model                 | `http://host.docker.internal:11434`     |
| `OLLAMA_MODEL`        | Name of the Ollama embedding model            | `nomic-embed-text`                      |

> ⚠️ **Note**:  If ollama is running in host machine and vector database, i.e., WEAVIATE, in docker, then we use `http://host.docker.internal:11434`, which is also the default value. However, if both are running in docker in the same host, use `http://localhost:11434 `.
#### 🧵 Optional Integrations

| Variable               | Description                                                                | Default           |
|------------------------|----------------------------------------------------------------------------|-------------------|
| `ENABLE_WEIGHTSANDBIAS` | Enable [Weights & Biases](https://wandb.ai/site) monitoring (`true/false`) | `false`           |
| `ENABLE_MLFLOW`        | Enable [MLflow](https://mlflow.org/) logging (`true/false`)                | `false`           |
| `MLFLOW_TRACKING_URL`  | MLflow tracking server URL                                                 | `http://localhost:5000` |
> ⚠️ **Note**: `WEAVIATE_API_KEY` is **required** for `structsense` to run. If it's not set, the system will raise an error.
>   For Weights & Biases you need to create a project and provide it's key.



```shell
# Example .env file

WEAVIATE_API_KEY=your_api_key
WEAVIATE_HTTP_HOST=localhost
WEAVIATE_HTTP_PORT=8080
WEAVIATE_HTTP_SECURE=false

WEAVIATE_GRPC_HOST=localhost
WEAVIATE_GRPC_PORT=50051
WEAVIATE_GRPC_SECURE=false

WEAVIATE_TIMEOUT_INIT=30
WEAVIATE_TIMEOUT_QUERY=60
WEAVIATE_TIMEOUT_INSERT=120

OLLAMA_API_ENDPOINT=http://host.docker.internal:11434
OLLAMA_MODEL=nomic-embed-text

ENABLE_WEAVE=true
ENABLE_MLFLOW=true
MLFLOW_TRACKING_URL=http://localhost:5000
```
#### 🛠️ Minimum Required Environment Variables

Below are the **minimum required environment variables** to run `structsense`.  
This configuration assumes all other optional variables will use their default values.

In this minimal setup:

- 🚫 **Weights & Biases** is disabled  
- 🚫 **MLflow tracking** is disabled  
- 🚫 **Knowledge source integration** is disabled  
- 📦 As a result, **no vector database** (e.g., Weaviate) is used

```shell 
ENABLE_WEIGHTSANDBIAS=false
ENABLE_MLFLOW=false
ENABLE_KG_SOURCE=false 
OLLAMA_API_ENDPOINT=http://host.docker.internal:11434
OLLAMA_MODEL=nomic-embed-text 
```
---

### 📄 YAML Configuration
In order to run `structsense` you need 5 YAML configuration files.
- The first is the `agent configuration`.
  - The agent configuration. You can define as many agents as you want, we process it dynamically.
    - Example agent configuration.
      ```yaml 
      agents:
        - id: extractor_agent
          output_variable: extracted_info
          role: >
            [Entity Extraction Agent]
          goal: >
            Perform Named Entity Recognition (NER) or entity extraction on {input_data} and return structured JSON output.
          backstory: >
            You are an AI assistant specialized in information extraction for a specific domain. 
            Your expertise includes identifying and classifying entities relevant to the task, such as concepts, locations, people, or other domain-specific items. 
            You respond strictly in structured JSON to ensure compatibility with downstream systems.
          llm:
            model: openrouter/openai/gpt-4o-2024-11-20
            base_url: https://openrouter.ai/api/v1
            frequency_penalty: 0.1
            temperature: 0.7
            seed: 53
            api_key: YOUR_API_KEY_HERE  # Replace with your actual API key or use env var
      
          - id: alignment_agent
            output_variable: aligned_entities
            role: >
              [Concept Alignment Agent]
            goal: >
              Align extracted entities from {extracted_info} with domain-specific ontologies or schema models and return structured JSON.
            backstory: >
              You are an AI assistant with expertise in linking extracted terms to formal knowledge representations such as taxonomies, schemas, or ontologies. 
              Your responses help enrich and normalize the raw extracted data for semantic interoperability.
            llm:
              model: openrouter/openai/gpt-4o-2024-11-20
              base_url: https://openrouter.ai/api/v1
              frequency_penalty: 0.1
              temperature: 0.7
              seed: 53
              api_key: YOUR_API_KEY_HERE
    
          - id: judge_agent
            output_variable: reviewed_output
            role: >
              [Judgment & Scoring Agent]
            goal: >
              Evaluate the {aligned_entities} based on predefined criteria and return structured feedback and scores in JSON format.
            backstory: >
              You are an evaluation-focused AI agent that reviews entity alignment or extraction quality based on accuracy, consistency, or relevance. 
              You assign a confidence score (e.g., from 0 to 1) and provide justification or flags where applicable.
            llm:
              model: openrouter/openai/gpt-4o-2024-11-20
              base_url: https://openrouter.ai/api/v1
              frequency_penalty: 0.1
              temperature: 0.7
              seed: 53
              api_key: YOUR_API_KEY_HERE 
        ```
    - In the YAML file: 
        - **ID**: Unique identifier
        - **Goal**: Task to be performed
        - **LLM config**: Model, base URL, temperature, etc.
        - **Backstory**: Background knowledge the agent leverages
        - **Output variable**: Result name for the next agent/task
      
    For further details, refer to [Role-Goal-Backstory](https://docs.crewai.com/guides/agents/crafting-effective-agents#core-principles-of-effective-agent-design)
- The second is the `task configuration`.
 - Task configuration allows you to describes the tasks for the agent. 
    - Example task configuration.
 
## 📦 Installation
Install this package via :

```sh
pip install structsense
```

Or get the newest development version via:

```sh
pip install git+https://github.com/sensein/ner_framework.git
```