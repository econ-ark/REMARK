# PortfolioChoiceBlogPost

### Reproduce the results and figures

You can reproduce all of the results of this REMARK on any computer that can run [docker](https://en.wikipedia.org/wiki/Docker_(software)).

To install docker locally follow the [guide](https://github.com/econ-ark/econ-ark-tools/tree/master/Virtual/Docker#install-docker-desktop-macos-and-windows), and launch it so that it is running in the background on your computer.

(Instructions below will work in the terminal on a Mac or Linux machine; Windows users will need to [install bash](https://itsfoss.com/install-bash-on-windows/))
- Then [clone](https://www.toolsqa.com/git/git-clone/) the REMARK repository locally

```
$ git clone https://github.com/econ-ark/REMARK
```
- To make sure figures are reproduced on the correct path, make sure you are in the PortfolioChoiceBlogPost directory
```
$ cd REMARKs/PortfolioChoiceBlogPost
```

As a sanity check, you can confirm the present working directory using

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
