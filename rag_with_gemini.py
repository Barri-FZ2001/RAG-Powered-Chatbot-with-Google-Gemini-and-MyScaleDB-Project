from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
import pandas as pd
import clickhouse_connect


#### Load pdf file
loader = PyPDFLoader("Employee-Handbook.pdf")
pages = loader.load_and_split()
pages = pages[4:]  # Skip the first few pages as they are not required
text = "\n".join([doc.page_content for doc in pages])

#### Create chunks 
text_splitter= RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
    is_separator_regex=False
)

chunks = text_splitter.create_documents([text])
for i, chunk in enumerate(chunks):
    chunk.metadata={"doc_id":i}

#### Create embeddings 
my_api_key="YOUR_API_KEY"
genai.configure(api_key=my_api_key)
def get_embeddings(text):
    model='models/embedding-001'
    embedding=genai.embed_content(
        model=model,
        content=text,
        task_type='retrieval_document'
    )
    return embedding['embedding']

content_list = [doc.page_content for doc in chunks]
embeddings=[get_embeddings(doc.page_content) for doc in chunks]

df=pd.DataFrame({
    "page_content":content_list,
   "embeddings": embeddings})

# Insert data into myScaleDB
client = clickhouse_connect.get_client(
    host='host',
    port=443,
    username='username',
    password='passwd'
)
# Create a table with the name 'handbook'
client.command("""
    CREATE TABLE default.handbook (
        id Int64,
        page_content String,
        embeddings Array(Float32),
        CONSTRAINT check_data_length CHECK length(embeddings) = 768
    ) ENGINE = MergeTree()
    ORDER BY id
""")
# Insert data in batches
batch_size=10
num_batches=len(df)//batch_size

for i in range(num_batches):
    start_idx=i*batch_size
    end_idx=start_idx+batch_size
    batch_data=df.iloc[start_idx:end_idx]
    client.insert("default.handbook",batch_data.to_records(index=False).tolist(), column_names=batch_data.columns.tolist())
    print(f"Batch {i+1} inserted successfully.")

client.command("""
    ALTER TABLE default.handbook
    ADD VECTOR INDEX vec_index embeddings
    TYPE MSTG
""")

# Retrieve relevant documents 
def retrieve_docs(query):
    embedding_query = get_embeddings(query)
    results=client.query(f"""
        SELECT page_content,
        distance(embeddings, {embedding_query}) as dist FROM default.handbook ORDER BY dist LIMIT 3
    """)
    relevant_results=[]
    for result in results.named_results():
        relevant_results.append(result['page_content'])
    return relevant_results

def make_rag_prompt(query, relevant_passage):
    relevant_passage = ' '.join(relevant_passage)
    prompt = (
        f"You are a helpful and informative chatbot that answers questions using text from the reference passage included below. "
        f"Respond in a complete sentence and make sure that your response is easy to understand for everyone. "
        f"Maintain a friendly and conversational tone. If the passage is irrelevant, feel free to ignore it.\n\n"
        f"QUESTION: '{query}'\n"
        f"PASSAGE: '{relevant_passage}'\n\n"
        f"ANSWER:"
    )
    return prompt

# Generate response
def generate_response(user_prompt):
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(user_prompt)
    return answer.text

def generate_answer(query):
    relevant_text = retrieve_docs(query)
    text = " ".join(relevant_text)
    prompt = make_rag_prompt(query, relevant_passage=relevant_text)
    answer = generate_response(prompt)
    return answer
answer = generate_answer(query="can you explain me what the data security guideline said")
print(answer)
