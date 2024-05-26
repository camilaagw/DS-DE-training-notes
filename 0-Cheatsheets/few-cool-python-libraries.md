### getpass
Sample usage (in a Jupyter Notebook / Google Collab):
```python
import getpass, os
credential_names = ["GCP_PROJECT_ID", "ASTRA_DB_ENDPOINT", "ASTRA_DB_TOKEN"]

for credential in credential_names:
  if credential not in os.environ:
     os.environ[credential] = getpass.getpass("Provide your " + credential)
```

### rich
Friendly Debugging with Rich:
```
from rich import inspect

fig, ax = plt.subplots()
plt.close()

inspect(fig)
# to see the methods of the object
# inspect(fig, methods=True)
inspect(ax)
```
