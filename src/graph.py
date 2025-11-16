import operator
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, BaseMessage
from langgraph.graph.message import add_messages
import os
from pydantic import BaseModel, Field
from typing import List, Literal, TypedDict, Annotated
from langchain.agents import create_agent

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langsmith import Client
from setuptools import Command

import json

intro_writing_example = "Sometimes a job feels like the perfect fit: that’s exactly how I felt when I saw this Registrar position for King’s College. As a previous College resident and current blue card holder with extensive experience in university administration, teaching, and client-facing roles, I’ve built the skills and insight to transition smoothly into this position and deliver work of the highest quality."

technical_writing_example = "Colleges bring together people from all backgrounds, requiring a balance of technical and communication skills to support their diverse needs. Through my university degrees in teaching and work experience in tertiary education, I’ve had 4 years of living in university residential environments. Additionally, I’ve also had one year of experience as a residential assistant. As such, I’m strongly familiar with its admission processes, duty of care, and confidentiality. I have at times acted as an administrative assistant within my home university’s sporting facility, which familiarised me with event and tour organisation. My role as a sales assistant and receptionist has given me hands-on experience with databases, client management software, and booking systems. This background has equipped me to use administrative tools confidently and adapt quickly to StarRez and PortalX, ensuring a smooth transition into a registrar role with King’s."

soft_skills_writing_example = "In front-facing roles, strong communication is just as vital as technical ability: this is where I truly excel. My background as a university admin assistant, language teacher, and sales assistant has honed my verbal and written communication to a high standard. It’s given me the confidence and humility to effectively communicate with students, their parents and other relevant stakeholders in an ethical, respectful manner. My sales skills could similarly make me an indispensable contributor to the marketing and communications team. I can confidently work alongside King's College’s network to deliver the best service as a registrar."

conclusion_writing_example = "Having experiences as a college resident and assistant, admin assistant and teacher, life at university holds a special place in my heart. Stepping into a role within the registrar is a natural next step in my career: my background has given me the skills, knowledge, and temperament to be an effective, collaborative team member. I’m confident I can contribute positively to the team’s culture and provide high-quality support to students and stakeholders of all backgrounds. I would be grateful for the opportunity to discuss my application further and the possibility of becoming more involved with the community."

load_dotenv("/home/david/Git/JobScraper/backend/.env")

file_path = '/home/david/Git/JobScraper/backend/src/resume.txt'  # Replace with the actual path to your .txt file
resume = ""

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        resume = file.read()
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")


client = Client()
extraction_model = init_chat_model("gpt-5-nano", model_provider="openai", temperature=0)
writing_model = init_chat_model("gemini-2.5-flash", model_provider="google-genai", temperature=0.85)
editing_model = init_chat_model("gemini-2.5-flash", model_provider="google-genai", temperature=0.4)

class State(MessagesState):
    job_description: str
    important_information: List[str]
    required_technical_skills: List[str]
    required_soft_skills: List[str]
    resume_technical_skills: dict
    resume_soft_skills: dict
    technical_skills_paragraph: str
    soft_skills_paragraph: str
    opening_paragraph: str
    closing_paragraph: str
    full_cover_letter: str

class OutputState(TypedDict):
    full_cover_letter: str

def input_node(state: State) -> State:
    """Do input parsing, find the required technical and soft skills from the job description"""

    print(state)
    print(state["messages"])

    job_description = state["messages"][-1].content

    prompt = client.pull_prompt("job_description_input")
    prompt = prompt.invoke({"context": job_description})
    response = extraction_model.invoke(prompt).content

    json_output = json.loads(response)
    
    state["job_description"] =  job_description
    state["important_information"] = json_output["important_information"]
    state["required_technical_skills"] = json_output["required_technical_skills"]
    state["required_soft_skills"] = json_output["required_soft_skills"]
  
    
    return state


def technical_skills_extractor_node(state: State) -> State:
    """Extract technical skills from job description"""
    
    prompt = client.pull_prompt("resume_technical_skills_extractor")
    prompt = prompt.invoke({"required_technical_skills": state["required_technical_skills"], "resume": resume})
    response = extraction_model.invoke(prompt).content

    json_output = json.loads(response)
    state["resume_technical_skills"] = json_output

    return state


def technical_skills_summariser_node(state: State) -> State:
    """Summarise technical skills from job description"""
    
    prompt = client.pull_prompt("resume_technical_skills_summariser")
    prompt = prompt.invoke({"important_information": state["important_information"], "resume_technical_skills": state["resume_technical_skills"], "writing_example": technical_writing_example})
    response = writing_model.invoke(prompt).content

    state["technical_skills_paragraph"] = response

    return state


def soft_skills_extractor_node(state: State) -> State:
    """Extract soft skills from job description"""
    
    prompt = client.pull_prompt("resume_soft_skills_extractor")
    prompt = prompt.invoke({"required_soft_skills": state["required_soft_skills"], "resume": resume})
    response = extraction_model.invoke(prompt).content

    json_output = json.loads(response)
    state["resume_soft_skills"] = json_output

    return state


def soft_skills_summariser_node(state: State) -> State:
    """Summarise soft skills from job description"""
    
    prompt = client.pull_prompt("resume_soft_skills_summariser")
    prompt = prompt.invoke({"resume_soft_skills": state["resume_soft_skills"], "important_information": state["important_information"], "writing_example": soft_skills_writing_example})
    response = writing_model.invoke(prompt).content

    state["soft_skills_paragraph"] = response

    return state


def opening_paragraph_node(state: State) -> State:
    """Generate opening paragraph for cover letter"""
    
    prompt = client.pull_prompt("resume_opening_paragraph_generator")
    prompt = prompt.invoke({"important_information": state["important_information"], "technical_skills_paragraph": state["technical_skills_paragraph"], "soft_skills_paragraph": state["soft_skills_paragraph"], "writing_example": intro_writing_example})
    response = writing_model.invoke(prompt).content

    state["opening_paragraph"] = response

    return state


def closing_paragraph_node(state: State) -> State:
    """Generate closing paragraph for cover letter"""

    prompt = client.pull_prompt("resume_closing_paragraph_generator")
    prompt = prompt.invoke({"important_information": state["important_information"], "technical_skills_paragraph": state["technical_skills_paragraph"], "soft_skills_paragraph": state["soft_skills_paragraph"], "writing_example": conclusion_writing_example})
    response = writing_model.invoke(prompt).content

    state["closing_paragraph"] = response

    return state
    

def output_node(state: State) -> State:
    """Generate full cover letter"""

    paragraphs = state["opening_paragraph"] + "\n\n" + state["technical_skills_paragraph"] + "\n\n" + state["soft_skills_paragraph"] + "\n\n" + state["closing_paragraph"]

    prompt = client.pull_prompt("resume_full_cover_letter_generator")
    prompt = prompt.invoke({"paragraphs": paragraphs, "job_description": state["job_description"]})
    response = editing_model.invoke(prompt).content

    return OutputState(full_cover_letter=response)


# Build workflow
builder = StateGraph(State, input_type=MessagesState, output_type=OutputState)

# Add nodes
builder.add_node("input_node", input_node)
builder.add_node("technical_skills_extractor_node", technical_skills_extractor_node)
builder.add_node("technical_skills_summariser_node", technical_skills_summariser_node)
builder.add_node("soft_skills_extractor_node", soft_skills_extractor_node)
builder.add_node("soft_skills_summariser_node", soft_skills_summariser_node)
builder.add_node("opening_paragraph_node", opening_paragraph_node)
builder.add_node("closing_paragraph_node", closing_paragraph_node)
builder.add_node("output_node", output_node)


builder.add_edge(START, "input_node")

builder.add_edge("input_node", "technical_skills_extractor_node")
builder.add_edge("technical_skills_extractor_node", "technical_skills_summariser_node")
builder.add_edge("technical_skills_summariser_node", "soft_skills_extractor_node")
builder.add_edge("soft_skills_extractor_node", "soft_skills_summariser_node")
builder.add_edge("soft_skills_summariser_node", "opening_paragraph_node")
builder.add_edge("opening_paragraph_node", "closing_paragraph_node")
builder.add_edge("closing_paragraph_node", "output_node")
builder.add_edge("output_node", END)


graph = builder.compile()



# job_description = """   

# Casual Security Officer Evergreen

# locations
# St Lucia Campus
# time type
# Part time
# posted on
# Posted 30+ Days Ago
# job requisition id
# R-35409
# Property and Facilities Division 

# Opportunity to join our casual pool of security officers

# Casual position at HEW Level 3. The hourly rate (incl. 25% casual loading) is $42.93 per hour plus 11% Superannuation. 

# Based at the picturesque St Lucia Campus 

# About This Opportunity   
# The Security Officers at The University of Queensland (UQ) are to provide services designed to ensure the protective security of all University buildings and grounds, staff, students and visitors on a relevant campus. A major challenge to this position is to provide a professional service to the University of Queensland and deal effectively with students, staff and public, while maintaining good public relations, as set down in the Standards of Operation and Standards of Personal Conduct. 

 

# Key responsibilities but not limited to: 

# Under supervision attend all response to alarms, medical emergencies, fire emergencies, or other matters as they arise. 

# Operate the Central Security Monitoring Station (CSMS). 

# Respond to enquiries/complaints of public/staff/students. 

# Lock, patrol and provide access for authorised personnel to University buildings. 

# Bring to the immediate notice of the Security Supervisor any matter of priority or emergency. 

# Conduct enforcement of parking and traffic regulations and control. 

# Conduct preliminary investigations into all incidents as directed.

# Prepare full, detailed and accurate reports by the end of each shift of all incidents that occurred during the shift using standard formats. 

# Perform other duties as reasonably directed by the Manager Security (MS), the Deputy Manager Security (DMS) or the Security Supervisor (SS) or delegate. 

 

# About UQ 
# As part of the UQ community, you’ll have the opportunity to work alongside the brightest minds, who have joined us from all over the world.  

 

# Everyone here has a role to play. As a member of our professional staff cohort, you will be actively involved in working towards our vision of a better world. By supporting the academic endeavour across teaching, research, and the student life, you’ll have the opportunity to contribute to activities that have a lasting impact on our community. 

 
# Please Note: All successful applicants will need to have full availability to attend 6 weeks full-time training. 

# About You 
# Current Queensland Certificate II in Security Operations that includes Security Officer Unarmed, Crowd Control, Bodyguard and Monitoring modules 

# Current St John, Red Cross or Qld Ambulance First Aid Certificate

# Current Queensland Manual Open Drivers Licence 

# Well-developed knowledge of security methodologies and concepts as applied to the Security industry 

# Sound interpersonal skills, including the ability to communicate effectively with a large variety of individuals, both internal and external to the University 

# Demonstrated proficiency with security software and programs and the ability to produce and analyse reports in a timely manner 

# Demonstrated ability to work with minimum supervision and to efficiently and effectively organise the work under your jurisdiction within strict deadlines 

# The ability to exercise tact and restrain in the face of direct provocation or occasional unfair criticism 

# Demonstrated experience effectively dealing with emergency situations arising during the course of duty, such as student demonstrations 

# The successful candidate may be required to complete background checks including, right to work in Australia and criminal history.  

#  Questions?   
# For more information about this opportunity, please contact Andrew Barling (Manager, Security Operations & Crime Prevention) on a.barling@pf.uq.edu.au 

# For application queries, please contact talent@uq.edu.au stating the job reference number in the subject line.   

 

# Want to Apply?  
# All applicants must upload the following documents in order for your application to be considered: 

# Cover letter addressing the ‘About You’ section   

# Resume  

 

# Other Information  
# As a casual employee, and in line with Fair Work, please note that there is no commitment to ongoing work in this position.  

# UQ is committed to a fair, equitable and inclusive selection process, which recognises that some applicants may face additional barriers and challenges which have impacted and/or continue to impact their career trajectory. Candidates who don’t meet all criteria are encouraged to apply and demonstrate their potential. The selection panel considers both potential and performance relative to opportunities when assessing suitability for the role.  

 

# We know one of our strengths as an institution lies in our diverse colleagues. We're dedicated to equity, diversity, and inclusion, fostering an environment that mirrors our wider community. We're committed to attracting, retaining, and promoting diverse talent. Reach out to talent@uq.edu.au for accessibility support or adjustments. 

 

# Applications will be reviewed on an ongoing basis. (R-35409) 

# #LI-DNI 
# """

# input_state: InputState = {
#     "job_description": job_description
# }

# output_state = graph.invoke(input_state)