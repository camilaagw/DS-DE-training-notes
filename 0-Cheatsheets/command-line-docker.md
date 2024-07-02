### Basic Docker commands

TODO 

### Colima: One-stop Docker solution

If you are in search for a solution alternative to Docker Desktop, Colima is the solution. It is lightweight, fast, and open-source docker runtime without licensing restrictions.

* Install Colima: `brew install colima`

* Adjust CPU, memory and disk limits as needed:

```bash
# Intel
$ colima start --vm-type=vz --cpu 6 --memory 16
 
# Apple
$ colima start --arch aarch64 --vm-type=vz --vz-rosetta --cpu 8 --memory 16
```

After creation, you can manage the VM with the commands `colima list`, `colima status`, `colima ssh`, `colima stop`, `colima start`, `colima restart`, `colima delete`, ..., etc.

### Lima: Going beyond Colima

In case you want more control on your setup or manage several linux VMs, you can use Lima directly, in addition to Colima or not.
```bash
$ limactl create --vm-type=vz --network=vzNAT --rosetta # Use the VZ hypervisor on Apple silicon
$ limactl start default
...
$ lima nerdctl run -it remote-docker.artifactory.<company>.com/ubuntu:latest
```



