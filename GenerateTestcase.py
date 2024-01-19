import streamlit as st
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import langchain
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import os
from langchain.document_loaders import PyPDFDirectoryLoader
import fitz  # PyMuPDF
import PyPDF2
from langchain.llms import OpenAI


openai_api_key  = ""
embeddings=OpenAIEmbeddings(openai_api_key="")
#Vector searchDb in pinecone
pinecone.init(api_key="",environment="gcp-starter")


text_to_add = """"""  


question_context="""Above is a question and knowledge context of Toyota multimedia and Navigation unit ,
        """



def add_text_to_pdf(input_pdf_path, newdir,newfilename, text_to_add):
    # Open the existing PDF file
    file_path_new = os.path.join(newdir, newfilename)
    pdf_document = fitz.open(input_pdf_path)
    page = pdf_document.new_page(width=500, height=800)
    text_annotation = page.insert_text((30, 30), text_to_add,fontsize=9)
    pdf_document.save(file_path_new)
    pdf_document.close()
    os.remove(input_pdf_path)
    return file_path_new

def save_pdf(dirname,filename):
    os.makedirs(dirname, exist_ok=True) 
    file_path = os.path.join(dirname,filename.name)
    with open(file_path, 'wb') as file:
            file.write(filename.read())
            st.success(f"File saved to: {file_path}") 
            return file_path    

def chunk_embedding(directory,index_name):
        #reading directory and chunking pdf file
        dir_loader=PyPDFDirectoryLoader(directory)   
        documents=dir_loader.load()   
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        chunks=text_splitter.split_documents(documents) 

        # only create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(        
            name=index_name,
            dimension=1536,
            metric='cosine')

        index=Pinecone.from_documents(chunks,embeddings,index_name=index_name)
        return index

def  process_query(question,index_name):  
        pinecone.init(api_key="",environment="gcp-starter")
        index = pinecone.Index(index_name)
        query_text = question
        query_embedd = embeddings.embed_query(query_text)
        res = index.query(query_embedd, top_k=5, include_metadata=True)
        matching_results = [x['metadata']['text'] for x in res['matches']] 
        #print(matching_results[0])
        return matching_results[0]

def get_final_answer(question,queryResult):
     matching_results =" RELEVENT KNOWLEDGE CONTEXT FROM CHAPTER as mentioned below: \n"+ queryResult
     prompt= question+ matching_results + "\n"*3 + "\n"*3 + question_context
     #prompt=prompt+"\n"+"If the information provided does not contain the answer,mention do not have the answer from the context provided"
     if st.button("Answer"):
             response = get_openai_resp(prompt)
             st.text("ChatGPT API reply:")
             st.write(response)






def get_openai_resp(prompt):
    # using OpenAI's Completion module that helps execute
    # any tasks involving text
    # Then, you can call the "gpt-3.5-turbo" model
    model_engine = "gpt-3.5-turbo"
    openai.api_key  = ""
    response = openai.ChatCompletion.create(
        model=model_engine,
        temperature=0.5,
        messages=[{"role": "user", "content": prompt}])
    output_text = response['choices'][0]['message']['content']
    print("ChatGPT API reply:", output_text)
    return output_text


         
def main():
    st.title("Test case cration")   
    #uploaded_pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    #if uploaded_pdf_file is not None:
        #Saving uploaded pdf into directory uploads
        #file_path_original= save_pdf("uploads",uploaded_pdf_file)

        #Adding Bloom texanomy information to pdf and creating new pdf with name modified.pdf
        #file_path_modified=add_text_to_pdf(file_path_original,"uploads","modified.pdf", text_to_add)

      
        #chunking pdf,making embedding and upserting in pinecone db
        #index=chunk_embedding("uploads","semanticsearch")

    question = st.text_input("Enter your question")
        
    if question:
           queryResult=process_query(question,'semanticsearch')
           get_final_answer(question,queryResult)
 
          





if __name__ == "__main__":
    main()
 