import os
import re
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Inisialisasi Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
googleai_key = os.environ.get("GOOGLEAI_KEY")
openai_key = os.environ.get("OPENAI_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Inisialisasi embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", openai_api_key=openai_key
)


def sanitize_documents(documents):
    """Membersihkan karakter null byte dan karakter non-printable lainnya."""
    cleaned_docs = []
    for doc in documents:
        content = doc.page_content

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")

        if isinstance(content, str):
            # Hapus null byte dan karakter non-printable
            content = re.sub(
                r"[\x00-\x1F\x7F]", "", content
            )  # termasuk \x00 sampai \x1F (control chars)
            content = content.encode("utf-8", errors="ignore").decode(
                "utf-8", errors="ignore"
            )

        doc.page_content = content
        cleaned_docs.append(doc)

    return cleaned_docs


def process_load_pymupdf(path, table_name):
    loader = PyMuPDFLoader(path)
    documents = loader.load()
    documents = sanitize_documents(documents)

    # Potong dokumen dan upload tiap chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # vector_store = SupabaseVectorStore.from_documents(
    #     docs,
    #     embeddings,
    #     client=supabase,
    #     table_name=table_name,
    #     query_name="match_documents",
    #     chunk_size=1000,
    #     batch_size=25
    # )

    # Upload dokument per batch
    batch_size = 100
    counter = 0
    for i in range(0, len(docs)):
        if counter + batch_size >= len(docs):
            batch_docs = docs[counter:len(docs)]
        else:
            batch_docs = docs[counter:counter + batch_size]

        counter += batch_size

        SupabaseVectorStore.from_documents(
            batch_docs,
            embedding=embeddings,
            client=supabase,
            table_name=table_name,
            query_name="bpka_match_documents"
        )
        print(f'batch {i} completed')
        if counter + batch_size >= len(docs):
            break


list_doc = os.listdir('./BPKA_Data')
for i in range(11, 15):
    try:
        process_load_pymupdf(f"BPKA_Data/{list_doc[i]}", "bpka_vector")
        print(f"[LOADED] file {list_doc[i]} berhasil di-load.")
    except Exception as e:
        print(f"[FAIL] File {list_doc[i]} gagal diload: {e}")
# File yang diupload sekitar 15 dokumen
print("Completed")
    