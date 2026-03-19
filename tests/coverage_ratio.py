import requests, os
from dotenv import load_dotenv
from neo4j import GraphDatabase
load_dotenv()

# 1. CourtListener — total opinions cited in 4th Amendment cases 2015-2025
token = os.getenv('COURTLISTENER_TOKEN')
headers = {'Authorization': f'Token {token}'}
params = {
    'type': 'o',
    'q': 'fourth amendment search seizure',
    'filed_after': '2015-01-01',
    'filed_before': '2025-12-31',
    'court': 'ca1 ca2 ca3 ca4 ca5 ca6 ca7 ca8 ca9 ca10 ca11 cadc',
    'page_size': 1,
}
r = requests.get('https://www.courtlistener.com/api/rest/v4/search/', headers=headers, params=params)
print('CourtListener 4th Amendment opinions 2015-2025:', r.json().get('count', 'error'))

# 2. Neo4j — total stub nodes (cited cases not in corpus)
driver = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')))
with driver.session() as s:
    stub_count = s.run('MATCH (c:Case {stub: true}) RETURN count(c) AS n').single()['n']
    full_count = s.run('MATCH (c:Case {stub: false}) RETURN count(c) AS n').single()['n']
    total = s.run('MATCH (c:Case) RETURN count(c) AS n').single()['n']
    print(f'Neo4j full nodes: {full_count}')
    print(f'Neo4j stub nodes: {stub_count}')
    print(f'Neo4j total nodes: {total}')
driver.close()