- Check current context:
  $ `kubectl config get-contexts`

- Set context:
  $ `kubectl config use-context <context>`

- Set namespace:
  $ `kubectl config set-context --current --namespace=<namespace>`

- List running pods:
  $ `kubectl get pods --sort-by=.metadata.creationTimestamp -l org=<org>,env=<env>`
  # Optionally:
  #  `--field-selector status.phase=Running`

- Port-forwarding:
  With eg. <id>=service/postgres, <local>=2067, <remote>=5432
  $ `kubectl port-forward <id> <local>:<remote>`

- Check pods:
  $ `kubectl --context <contex> --namespace <namespace> get pods`
  # Optionally add:
  # ` | grep <pod-ids>`

- Stream logs of a running pod's container:
  $ `kubectl logs -p -c <container> <pod>`

- Restarting a deployment:
  $ `kubectl rollout restart deployment <deployment-name>`

- Execute an interactive shell within a specific deployment:
  $ `kubectl exec -it deployment/<deployment-name> -- /bin/bash`
