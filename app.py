from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate, set_global_tokenizer
from transformers import AutoTokenizer
from llama_index.core.memory import ChatMemoryBuffer
import gradio as gr

# Define the data directory for loading documents
DATA_DIR = "docs"

# Initialize LLM
llm = Ollama(model="llama2", request_timeout=180.0)

# Initialize HuggingFace Embedding Model for Vectorization
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Set the global tokenizer to use the tokenizer from HuggingFace for encoding inputs
set_global_tokenizer(
    AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
)

# Load documents from the data directory into the Vector Store Index
documents = SimpleDirectoryReader(DATA_DIR).load_data()

# Create Vector Store Index with HuggingFace Embedding
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Create Prompt Template for Text-based Q&A
chat_text_qa_msgs = [
    (
        "user",
        """You are a Q&A assistant. For all other inquiries, your main goal is to provide answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, kindly advise the user to ask questions within the context of the document.
    Context:
    {context_str}
    Question:
    {query_str}
    """
    )
]
text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

# Initialize Chat Memory Buffer for Conversation Memory
memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

# Create Chat Engine with LLM
chat_engine = index.as_chat_engine(
    llm=llm,
    text_qa_template=text_qa_template,
    streaming=True,
    memory=memory,
    # https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/modules/
    chat_mode="condense_question"  # Chooses mode suit for your use case
)


### Gradio Interface ###

def chat_with_ollama(message, history):
    # debug print memory
    # print(memory.get_all())

    if history == []:
        print("# cleared history, resetting chatbot state")
        chat_engine.reset()

    streaming_response = chat_engine.stream_chat(message)
    response = ""
    for text in streaming_response.response_gen:
        response += text
        yield response


chatbot = gr.ChatInterface(
    chat_with_ollama, title="Document-Based Chatbot with LLM")

chatbot.launch()
# chatbot.launch(server_name="xx.xx.xx.xx", server_port=7860)  # set IP and port for deployment
