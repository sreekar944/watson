from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from genai.model import Credentials
from genai.schemas import GenerateParams
from genai.extensions.langchain.llm import LangChainInterface
from dotenv import load_dotenv
from os import environ
import genai.exceptions.genai_exception
def getAnswer(query):
    load_dotenv()

    loader = PyPDFDirectoryLoader('.')
    documents = loader.load_and_split()

    model_id = 'google/flan-ul2'
    params = GenerateParams(
        decoding_method='greedy',
        repetition_penalty=1.0,
        min_new_tokens=1,
        max_new_tokens=100
    )
    # credentials= Credentials("dRviLRDv6w3i-SUm2vcESYFdcUUNF5HGHgbJrakywER4",api_endpoint="https://us-south.ml.cloud.ibm.com")
    credentials = Credentials(environ['GENAI_KEY'], api_endpoint=environ['GENAI_API'])
    llm = LangChainInterface(credentials=credentials, model=model_id, params=params)

    chain = load_qa_chain(llm, chain_type='stuff')
    print("run completed")
    return chain.run(input_documents=documents[0:7], question=query)

# if __name__ == '__main__':
#     getAnswer("What is the rating of IBM?")