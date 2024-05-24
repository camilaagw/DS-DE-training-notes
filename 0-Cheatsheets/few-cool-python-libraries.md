### getpass
Sample usage (in a Jupyter Notebook / Google Collab):
```python
import getpass, os
credential_names = ["GCP_PROJECT_ID", "ASTRA_DB_ENDPOINT", "ASTRA_DB_TOKEN"]

for credential in credential_names:
  if credential not in os.environ:
     os.environ[credential] = getpass.getpass("Provide your " + credential)
```
