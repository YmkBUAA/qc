from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.nacfql import NACFQLAgent
from agents.nacfql_2 import NACFQL2Agent

agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    nacfql=NACFQLAgent,
    nacfql_2=NACFQL2Agent,
)
