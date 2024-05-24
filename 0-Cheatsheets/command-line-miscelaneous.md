Useful tools:
- `pbpaste`: Paste the output contents from your clipboard.
- `pbcopy`: Copy the output of a command right into your clipboard: `$ cat myfile.txt | pbcopy`.
- `jq`: Command-line JSON processor. Useful for processing API responses.

Handy commands:
- `du -sh ./* | sort -rh `: list the sizes of files and dirs in the current directory in human-readable format and sorts them in descending order based on their sizes.
- `for f in $(yarn application -list | grep MyApp | sed 's/ [ \t]*/ /g' | cut -d' ' -f1 ); do echo "Killing $f..."; yarn application -kill $f; done`: Find and kill specific running applications containing the name "MyApp". The `sed 's/ [ \t]*/ /g'` command replaces multiple spaces or tabs with a single space. The `cut -d' ' -f1` command extracts the first field (application ID) from the output. 
- `ps -u service-user --forest -F`: List running processes in a hierarchical tree format for the `service-user`.
- `for i in $(seq 0 9 | xargs -I {} date -d "2022-02-01 {} months" +%Y-%m); do ./my_script.sh $i ; done`: Execute a script with date argument, iterating over a range of 10 months.

Compress files:
- `zip -r mydir.zip mydir -x "*/.*"`: Zip a directory named `mydir`, excluding hidden files/directories.
- `tar -czvf file.tar.gz -C ./Dir .`: Create a compressed tar file named `file.tar.gz` by archiving the contents of the `Dir` directory. The `-c` option tells tar to create a new archive. The `-z` option uses gzip compression to reduce the file size. The `-v` option shows verbose output, listing all the files as they are added to the archive. The `-f` option specifies the output file name.

Uncompress files:
- `tar -xvf file.tar`: Untar a  file with `.tar` extension. `x` stands for extract, `v` for verbose (it will list all the files as they are extracted), and `f` specifies that the next argument is the file name of the archive.
- `tar -xzvf file.tar.gz`: Untar a file compressed using gzip (`.tar.gz` or `.tgz` extension). The `z` option tells tar to decompress the archive using gzip before extracting files.
- `tar -xjvf file.tar.bz2`: Untar a file compressed using bzip2 (`.tar.bz2` extension). The `j` option tells tar to decompress the archive using bzip2 before extracting files.



