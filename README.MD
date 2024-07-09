# LLM RAG Ollama Gradio ✅

Retrieval-Augmented Generation (RAG) with Large Language Model (LLM) using llama-index library and Ollama.

Ollama untuk enabler local local dengan easy setup

Library llama-index sebagai framework RAG, dengan SimpleDirectoryReader, membaca seluruh dokumen dalam folder yang ditentukan

Interface berupa chatbot powered by Gradio

## Install & run Ollama

Kenapa dalam contoh kali ini menggunakan Ollama?

- LLM local
- instalasi yang mudah, dibanding llm cpp
- service tersendiri yang terpisah dari program utama

Installasi: <https://ollama.com/download>

Run:

1. ollama serve
2. ollama pull llama2 (pilih dan sesuaikan model yang dipakai di app.py)

## Download this source

download zip and extract or simply clone and pull

### install requirements

> pip install -r requirements.txt

### run app

> python app.py

## Powered by

Ollama <https://ollama.com/>

llama-index <https://docs.llamaindex.ai/en/stable/>

Gradio <https://www.gradio.app/>
