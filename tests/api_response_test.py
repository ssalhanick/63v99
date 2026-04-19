import requests

text = """
In United States v. Garrett, 887 F.3d 452 (9th Cir. 2021), the court held that 
the plain view doctrine permits officers to seize any item in a vehicle during a 
routine traffic stop without a warrant.
"""

resp = requests.post("http://localhost:8000/check-citation", json={"text": text})
import json
print(json.dumps(resp.json(), indent=2))