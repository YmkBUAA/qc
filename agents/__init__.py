from agents.acfql import ACFQLAgent
from agents.acfql_mt import ACFQLMTAgent
from agents.acrlpd import ACRLPDAgent
from agents.nacfql import NACFQLAgent
from agents.nacfql_2 import NACFQL2Agent
from agents.nacfql_3 import NACFQL3Agent
from agents.acfql_v import ACFQLVAgent

agents = dict(
    acfql=ACFQLAgent,
    acfql_mt=ACFQLMTAgent,
    acrlpd=ACRLPDAgent,
    nacfql=NACFQLAgent,
    nacfql_2=NACFQL2Agent,
    nacfql_3=NACFQL3Agent,
    acfql_v=ACFQLVAgent,
)
