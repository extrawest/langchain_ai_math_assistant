import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chains import LLMMathChain, LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Text to Math Problem Solver", page_icon=":wave:", layout="wide")
st.title("Text to Math Problem Solver with Gemma2")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please enter your Groq API key")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", api_key=groq_api_key)

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(name='Wikipedia',
                      func=wikipedia_wrapper.run,
                      description="A tool for searching Wikipedia to find  the various info",
                      )
math_chain = LLMMathChain.from_llm(llm=llm)
calculator_tool = Tool(name='Calculator',
                       func=math_chain.run,
                       description="A tool for calculating math problems"
                       )

prompt = """
You are an agent tasked for solving users mathematical questions.
 Provide a solution and reasoning, display it point wise for the question below
 Question: {question}
 Answer:
 """

prompt_template = PromptTemplate(input_variables=["question"], template=prompt)

chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(name="Reasoning tool",
                      func=chain.run,
                      description="A tool for reasoning about math problems"
                      )

assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello, I am a math assistant. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.text_area("Question",
                        "I start with 8 oranges and 10 strawberries. I eat 3 oranges and give away 4 strawberries. "
                        "Then I buy a bunch of 6 bananas and 3 boxes of raspberries, with each box containing 20 "
                        "berries. How many total pieces of fruit do I have at the end?")

if st.button("Submit"):
    if question:
        with st.spinner("Generating response.."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_callback])
            print(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write('### Response:')
            st.success(response)
    else:
        st.warning("Please enter the question")
