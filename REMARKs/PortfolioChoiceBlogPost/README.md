 
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

- If you make any changes to the jupyter notebook you need to rerender the HTML document.
```
jupyter nbconvert --to html PortfolioChoiceBlogPost.ipynb
```



## Other Experiments (Removed to Shorten Blog Post)

The `ConsPortfolioModel` tool makes it easy to explore alternative assumptions about any element of the model.  For example, after the CGM paper was published, better estimates [became available](https://doi.org/10.1016/j.jmoneco.2010.04.003) about the degree and types of income uncertainty that consumers face at different ages.  The most important finding was that the degree of uncertainty in earnings is quite large for people in their 20s but falls sharply then flattens out at older ages.  

It seems plausible that this greater uncertainty in labor earnings could account for the fact that in empirical data young people have a low share of their portfolio invested in risky assets; economic theory says that an increase in labor income risk should [reduce your willingness to expose yourself to financial risk](https://www.jstor.org/stable/2951719).

But the figure below shows that even when we update the model to incorporate the improved estimates of labor income uncertainty, the model still says that young people should have 100 percent of their savings in the risky asset.


<center><big>
    Using Better Data on Income Risk By Age
    </big>
<center>
    <img src='figures/figure_Parameters_1940s_shocks/RShare_Means.png'>
</center>

<!-- ![Parameters_1940s_shocks](figures/figure_Parameters_1940s_shocks/RShare_Means.png) --> 
