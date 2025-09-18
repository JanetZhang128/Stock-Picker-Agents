from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool

# have structured output for the agent, here below are the schemas for the agents to use
from pydantic import BaseModel, Field

class TrendingCompany(BaseModel):
    """A trending company that is in the news and attracting attention"""
    name: str = Field(description="Company name")
    ticker: str = Field(description="Stock ticker symbol")
    reason: str = Field(description="Reason this company is trending in the news")

class TrendingCompanyList(BaseModel):
    """A list of trending companies"""
    companies: List[TrendingCompany] = Field(description="List of trending companies")

class TrendingCompanyResearch(BaseModel):
    """Detailed Research on a trending company"""
    name: str = Field(description="Company name")
    market_position: str = Field(description="Current market position and competitive analysis")
    future_outlook: str = Field(description="Future outlook and growth potential")
    investment_potential: str = Field(description="Investment potential and suitability for investment")

class TrendingCompanyResearchList(BaseModel):
    """A list of detailed research on trending companies"""
    research_list: List[TrendingCompanyResearch] = Field(description="List of detailed research on trending companies")

@CrewBase
class StockPicker():
    """StockPicker crew"""

    agent_config = "config/agent_config.yaml"
    crew_config = "config/crew_config.yaml"

    @agent
    def trending_company_finder(self) -> Agent:
        """Agent that reads news articles and identifies trending companies"""
        return Agent(
            config=self.agent_config['trending_company_finder'],
            tools=[SerperDevTool()],
            verbose=True,
        )
    
    @agent
    def financial_researcher(self) -> Agent:
        """Agent that researches trending companies"""
        return Agent(
            config=self.agent_config['financial_researcher'],
            tools=[SerperDevTool()],
            verbose=True,
        )

    @agent
    def stock_picker(self) -> Agent:
        """Agent that picks stocks based on the trending companies"""
        return Agent(
            config=self.agent_config['stock_picker'],
            verbose=True,
        )
    
    @task
    def find_trending_companies(self) -> Task:
        return Task(
            config=self.task_config['find_trending_companies'],
            output_pydantic=TrendingCompanyList,
        )

    @task
    def research_trending_companies(self) -> Task:
        return Task(
            config=self.task_config['research_trending_companies'],
            output_pydantic=TrendingCompanyResearchList,
        )
    
    @task
    def pick_best_company(self) -> Task:
        return Task(
            config=self.task_config['pick_best_company'],
        )
    
    @crew
    def stock_picker_crew(self) -> Crew:
        """Crew that picks the best company for investment"""

        manager = Agent(
            config=self.agent_config['manager'],
            allow_delegation=True,
        )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            manager=manager,
            verbose=True,
            process=Process.hierarchical
        )