import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0,groq_api_key=os.getenv("GROQ_API_KEY"),model_name="compound-beta")

    def extract_jobs(self,cleaned_text):
            prompt_extract=PromptTemplate.from_template(
                """
                ### SCRAPED TEXT FROM WEBSITE:
                {page_data}
                ### INSTRUCTION:
                The scraped text is from the career's page of a website.
                Your job is to extract the job postings and return them in JSON format containing 
                the following keys: 'role','experience','skills' and 'description'.
                Only return the valid JSON.
                ###VALID JSON(NO PREAMBLE):
                """
            )
            chain_extract = prompt_extract | self.llm
            res = chain_extract.invoke(input={'page_data': cleaned_text})
            try:
               json_parser = JsonOutputParser()
               res = json_parser.parse(res.content)
            except OutputParserException:
              raise OutputParserException("Context too big. Unable to parse jobs.")
            return res if isinstance(res,list) else [res]
#Prompt for writing email
    def write_mail(self,job,links):
        prompt_email = PromptTemplate.from_template(
            """ 
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:

            You are Chandra Shekar, a AI/ML Engineer at Conduent.  Reach out to HR to express interest in the Job Position at this Company. 
            With 5 years of experience in this Field/Relevant Skills, express that I am confident that my background aligns well with the requirements.txt of this role.
            if HR could consider referring me for this opportunity. 
            I'm enthusiastic about the possibility of contributing to this specific Company and would appreciate any support HR can offer in this process.
            Your job is to write a cold email to the HR regarding the job mentioned above and asking for the referral for this position.
            Also add all the links from the following links to showcase my Chandra shekar's portfolio:{link_list}
            Remember you are Chandra Shekar, AI/ML Engineer at Conduent.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content


if __name__=="__main__":
    print(os.getenv("GROQ_API_KEY"))