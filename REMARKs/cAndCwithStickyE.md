---
tags:
  - REMARK
  - Replication
location_url: https://github.com/llorracc/cAndCwithStickyE
name: cAndCwithStickyE
summary: 'Sticky Expectations and Consumption Dynamics'
abstract: ''
notebook:
authors:
  - Christopher Carroll
  - Edmund Crawley
  - Jiri Slacalek
  - Kiichi Tokuoka
  - Matthew White
DOI: 
---


# cAndCwithStickyE is a Reproduction 

This is a reproduction of the results in the paper "Sticky Expectations and Consumption Dynamics" by Carroll, Crawley, Slacalek, Tokuoka, and White. The [html version](http://econ.jhu.edu/people/ccarroll/papers/cAndCwithStickyE) contains links to resources including the PDF version, the presentation slides, a repo containing the paper and code, and related material

The root directory contains three files:

* `do_min.py` executes the representative agent version of the model 
* `do_mid.py` reproduces some of the core results of the paper 
* `do_all.py` reproduces all computational results in the paper

Any of these can be run using ipython from the command line, for example:

ipython do_min.py

In all cases, there should be a message that indicates how long it takes to run the code on a particular computer with a particular configuration.

