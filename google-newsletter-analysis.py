import os
from langchain.agents import Tool
from langchain.agents import load_tools
from crewai import Agent, Task, Process, Crew
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import Ollama

# API Keys
os.environ["SERPER_API_KEY"] = "serp-api-here"  # Replace with your API key
os.environ["OPENAI_API_KEY"] = "openai-api-key-here"  # Replace with your OpenAI API key
os.environ["GEMINI_API_KEY"] = "gemini-api-key-here"  # Replace with your Gemini API key

# Initialize LLMs
gpt4_llm = OpenAI(model="gpt-4", temperature=0.7, api_key=os.environ["OPENAI_API_KEY"])
claude_llm = ChatGoogleGenerativeAI(
    model="gemini-pro", verbose=True, temperature=0.3, google_api_key=os.environ["GEMINI_API_KEY"]
)
ollama_llm = Ollama(model="mistral")  # Using a local open-source model

# Tools
search = GoogleSerperAPIWrapper()
search_tool = Tool(
    name="Scrape google searches",
    func=search.run,
    description="Useful for when you need to ask the agent to search the internet",
)

# Loading Human Tools
human_tools = load_tools(["human"])

# Explorer Agent Details
explorer_role = "Senior Researcher"
explorer_goal = "Find and explore the most exciting projects and companies in the AI and machine learning space in 2024"
explorer_backstory = """You are an expert strategist that knows how to spot emerging trends and companies in AI, tech, and machine learning. 
You're great at finding interesting, exciting projects on LocalLLama subreddit. You turn scraped data into detailed reports with names
of the most exciting projects and companies in the AI/ML world. ONLY use scraped data from the internet for the report."""

explorer = Agent(
    role=explorer_role,
    goal=explorer_goal,
    backstory=explorer_backstory,
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=gpt4_llm,
)

# Writer Agent Details
writer_role = "Senior Technical Writer"
writer_goal = "Write engaging and interesting blog post about the latest AI projects using simple, layman vocabulary"
writer_backstory = """You are an expert writer on technical innovation, especially in the field of AI and machine learning. You know how to write in 
an engaging, interesting but simple, straightforward, and concise style. You present complicated technical terms to a general audience in a 
fun way by using layman words. ONLY use scraped data from the internet for the blog."""

writer = Agent(
    role=writer_role,
    goal=writer_goal,
    backstory=writer_backstory,
    verbose=True,
    allow_delegation=True,
    llm=claude_llm,
)

# Critic Agent Details
critic_role = "Expert Writing Critic"
critic_goal = "Provide feedback and criticize blog post drafts. Make sure that the tone and writing style is compelling, simple, and concise"
critic_backstory = """You are an expert at providing feedback to technical writers. You can tell when a blog text isn't concise,
simple, or engaging enough. You know how to provide helpful feedback that can improve any text. You ensure that the text 
stays technical and insightful by using layman terms."""

critic = Agent(
    role=critic_role,
    goal=critic_goal,
    backstory=critic_backstory,
    verbose=True,
    allow_delegation=True,
    llm=ollama_llm,
)

# Tasks
task_report_description = """Use and summarize scraped data from the internet to make a detailed report on the latest rising projects in AI. Use ONLY 
scraped data to generate the report. Your final answer MUST be a full analysis report, text only, ignoring any code or anything that 
isn't text. The report has to have bullet points with 5-10 exciting new AI projects and tools. Write names of every tool and project. 
Each bullet point MUST contain 3 sentences that refer to one specific AI company, product, model, or anything you found on the internet."""

task_blog_description = """Write a blog article with text only and with a short but impactful headline and at least 10 paragraphs. The blog should summarize 
the report on the latest AI tools found on LocalLLama subreddit. The style and tone should be compelling and concise, fun, technical but also use 
layman words for the general public. Name specific new, exciting projects, apps, and companies in the AI world. Don't 
write "**Paragraph [number of the paragraph]:**", instead start the new paragraph in a new line. Write names of projects and tools in **BOLD**.
ALWAYS include links to projects/tools/research papers. ONLY include information from LocalLLAma."""

task_critique_description = """The Output MUST have the following markdown format:
## [Title of post](link to project)
    - Interesting facts
    - Own thoughts on how it connects to the overall theme of the newsletter
    ## [Title of second post](link to project)
    - Interesting facts
    - Own thoughts on how it connects to the overall theme of the newsletter
Make sure that it does and if it doesn't, rewrite it accordingly."""

task_report = Task(description=task_report_description, agent=explorer)
task_blog = Task(description=task_blog_description, agent=writer)
task_critique = Task(description=task_critique_description, agent=critic)

# Crew
crew = Crew(
    agents=[explorer, writer, critic],
    tasks=[task_report, task_blog, task_critique],
    verbose=2,
    process=Process.sequential,
)

# Kickoff
result = crew.kickoff()

print("######################")
print(result)
