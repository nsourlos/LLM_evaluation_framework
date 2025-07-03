import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from openai import OpenAI

from src.llm_eval.config import excel_file_name, reranker_model_name, openai_api_key
from ..utils.paths import get_file_paths

_, _, custom_cache_dir, _, _ = get_file_paths(excel_file_name)

def initialize_vectorstore(list_of_questions, list_of_answers, embedding_model, custom_cache_dir=custom_cache_dir):
    """Initialize the embedding model and FAISS vectorstore for the dataset."""
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        cache_folder=custom_cache_dir,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"} #We might even use 'mps' for MacOS
    )
    
    # Create documents from Q&A pairs
    documents = [
        Document(
            page_content=question,
            metadata={"answer": answer}
        ) for question, answer in zip(list_of_questions, list_of_answers)
    ]
    
    # Create and save FAISS index
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Optionally save the index for later use
    # vectorstore.save_local("faiss_index")
    
    return vectorstore

def initialize_reranker(reranker_model_name=reranker_model_name):
    """Initialize a cross-encoder reranker model."""
    reranker = CrossEncoder(reranker_model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    return reranker

def rerank_retrieved_documents(query, similar_pairs, reranker, top_k=3):
    """Rerank the retrieved documents using a cross-encoder model."""
    # Prepare pairs for reranking
    pairs = [(query, pair['question'] + "\n" + pair['answer']) for pair in similar_pairs]
    
    # Get scores from reranker
    scores = reranker.predict(pairs)
    
    # Combine with original documents and sort by new scores
    for i, pair in enumerate(similar_pairs):
        pair['rerank_score'] = float(scores[i])
    
    # Sort by reranker score (descending)
    reranked_pairs = sorted(similar_pairs, key=lambda x: x['rerank_score'], reverse=True)

    # Return top_k pairs with highest reranker scores
    return reranked_pairs[:top_k]

def get_similar_qa_pairs(query, vectorstore, top_k=5):
    """Get the most similar Q&A pairs using FAISS."""
    # Search for similar documents
    similar_docs = vectorstore.similarity_search_with_score(query, k=top_k)

    #Format results
    similar_pairs = []
    for doc, score in similar_docs:
        similar_pairs.append({
            'question': doc.page_content,
            'answer': doc.metadata['answer'],
            'similarity': 1 - score # Convert distance to similarity score
        })

    return similar_pairs

def check_context_relevance(query, similar_pairs, judge_model, openai_api_key=openai_api_key):
    """Check if the retrieved context is relevant enough to use for RAG."""

    # query="What is the weather in Rotterdam now?" #With this query the context is not relevant

    # Construct prompt for the judge
    prompt = f"""Given a user question and retrieved similar Q&A pairs, determine if the context is relevant enough to be used for answering the question.
    Consider:
    1. Semantic similarity between the question and retrieved pairs
    2. Whether the context provides useful information for answering
    3. If using no context might be better than using potentially misleading context

    User Question: {query}

    Retrieved Q&A Pairs:
    {similar_pairs}

    Should this context be used for answering the question? Respond with only 'Yes' or 'No'.
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that determines context relevance."},
        {"role": "user", "content": prompt}
    ]
    
    #Use OpenAI to judge relevance
    client = OpenAI(api_key=openai_api_key)
    
    response = client.chat.completions.create(
        messages=messages,
        temperature=0,
        model="/".join(judge_model.split('/')[1:]),
        seed=42
    )

    print("Use context/RAG:",response.choices[0].message.content.strip().lower()) #Rotterdam query above returns 'no'
    with open(f"rag_judge_{'_'.join(judge_model.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
        col_file.write(f"Use context/RAG: {response.choices[0].message.content.strip().lower()} \n")
    
    return response.choices[0].message.content.strip().lower() == 'yes'

def format_context(similar_pairs):
    """Format the similar Q&A pairs into a context string."""
    context = "Here are some relevant previous Q&A pairs:\n\n"
    for pair in similar_pairs:
        context += f"Question: {pair['question']}\n"
        context += f"Answer: {pair['answer']}\n\n"
    return context.strip() 