import dotenv
import os

from langchain_core.exceptions import OutputParserException
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

dotenv.load_dotenv()


class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model="llama-3.1-8b-instant",
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data} 
            ### INSTRUCTION: 
            The scraped text is from the career's page of a website. Your job is to extract the job postings and return them in JSON format containing following keys: 'role', 'skills', 'experience' and 'description' .
            Only return the valid JSON.
            ### OUTPUT FORMAT:
            JSON format containing following keys: 'role', 'skills', 'experience' ,'description' (NO PREAMBLE)
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
            
            ### INSTRUCTION: You are Sumaya, a business development executive at ACME AI. ACME AI is an AI & Software 
            Consulting company dedicated to facilitating the seamless integration of business processes through 
            automated tools. Over our experience, we have empowered numerous enterprises with tailored solutions, 
            fostering scalability, process optimization, cost reduction, and heightened overall efficiency. Your job 
            is to write a cold email to the hiring manager of that company regarding the job mentioned above 
            describing the capability of ACME AI in fulfilling their needs. Also add the most relevant ones from the 
            following links to showcase ACME's portfolio: {link_list} Remember you are Sumaya, BDE at ACME AI. Do not 
            provide a preamble. ### EMAIL (NO PREAMBLE):
            
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content


if __name__ == "__main__":
    print("Hello")
