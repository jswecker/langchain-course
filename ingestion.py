import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader('mediumblog1.txt')
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=768)
    vector = embeddings.embed_query("hello, world!")
    print(f'Length of each embedding: {len(vector)}')

    print("ingesting")
    # PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'], ssl_ca_certs="/home/xmmgr/certs/parsons_forward_trust.crt")
    pc = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
        ssl_ca_certs="/home/xmmgr/certs/parsons_forward_trust.crt",
    )

    index = pc.Index(os.environ["INDEX_NAME"])

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )

    vectorstore.add_documents(texts)

