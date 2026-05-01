"""
detector/doctrine_check.py

Phase 5, Step 5.3e — Doctrine Ontology checking.

Provides functions to interact with the (:Doctrine) nodes in Neo4j.
Used by the cross_citation aggregator to add doctrinal coherence signals.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_doctrines(session, case_id: int) -> list[str]:
    """
    Fetch all doctrine IDs applied to a specific case.
    """
    result = session.run(
        """
        MATCH (c:Case {id: $id})-[:APPLIES_DOCTRINE]->(d:Doctrine)
        RETURN collect(d.id) AS doctrines
        """,
        id=case_id
    )
    record = result.single()
    return record["doctrines"] if record and record["doctrines"] else []


def get_shared_doctrines(session, id_a: int, id_b: int) -> list[str]:
    """
    Fetch doctrine IDs that are shared between two cases.
    """
    result = session.run(
        """
        MATCH (a:Case {id: $id_a})-[:APPLIES_DOCTRINE]->(d:Doctrine)
              <-[:APPLIES_DOCTRINE]-(b:Case {id: $id_b})
        RETURN collect(DISTINCT d.id) AS shared_doctrines
        """,
        id_a=id_a, id_b=id_b
    )
    record = result.single()
    return record["shared_doctrines"] if record and record["shared_doctrines"] else []
