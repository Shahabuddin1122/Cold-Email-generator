import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chain import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text):
    st.title("📧 Cold Mail Generator")
    url_input = st.text_input("Enter a Job recruitment URL:", value="https://jobs.nike.com/job/R-39383")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader(url_input, header_template={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/102.0.0.0 Safari/537.36',
            })
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            job = llm.extract_jobs(data)
            skills = job[0].get('skills', [])
            links = portfolio.query_links(skills)
            email = llm.write_mail(job, links)
            st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)
