# PortfolioChoiceBlogPost





#### Reproduce the results and figures

This REMARK requires docker to reproduce the results.
Make sure you have installed Docker and it's running in the background.
To install docker locally follow the [guide](https://github.com/econ-ark/econ-ark-tools/tree/master/Virtual/Docker#install-docker-desktop-macos-and-windows)


- Clone the REMARK repository locally

```
$ git clone https://github.com/econ-ark/REMARK
```
- To make sure figures are reproduced on the correct path, make sure you are in the PortfolioChoiceBlogPost directory
```
$ cd REMARKs/PortfolioChoiceBlogPost
```
as a sanity check, you can confirm the present working directory using

```
$ pwd
```
and it should give you something like
```
/path_to_clone_of_REMARK/REMARKs/PortfolioChoiceBlogPost
```

- Run the following bash script in the directory to reproduce all the figures against different parameters.
```
$ ./do_all_code.sh
```

- This will reproduce the figures in the figures directory.