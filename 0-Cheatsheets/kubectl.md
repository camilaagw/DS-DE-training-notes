Check current context: <br>
$ `kubectl config get-contexts`

Set context: <br>
$ `kubectl config use-context <context>`

Set namespace: <br>
$ `kubectl config set-context --current --namespace=<namespace>`

List running pods: <br>
$ `kubectl get pods --sort-by=.metadata.creationTimestamp -l org=<org>,env=<env>`<br>
Optionally add `--field-selector status.phase=Running`

Port-forwarding:<br>
With eg. `<id>=service/postgres`, `<local>=2067`, `<remote>=5432` <br>
$ `kubectl port-forward <id> <local>:<remote>`

Check pods:  <br>
$ `kubectl --context <contex> --namespace <namespace> get pods`  <br>
Optionally add ` | grep <pod-ids>`

Stream logs of a running pod's container:  <br>
$ `kubectl logs -p -c <container> <pod>`

Restarting a deployment:  <br>
$ `kubectl rollout restart deployment <deployment-name>`

Execute an interactive shell within a specific deployment:  <br>
$ `kubectl exec -it deployment/<deployment-name> -- /bin/bash`
