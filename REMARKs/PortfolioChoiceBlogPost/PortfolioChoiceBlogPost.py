# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.9
#   latex_envs:
#     LaTeX_envs_menu_present: true
#     autoclose: false
#     autocomplete: false
#     bibliofile: biblio.bib
#     cite_by: apalike
#     current_citInitial: 1
#     eqLabelWithNumbers: true
#     eqNumInitial: 1
#     hotkeys:
#       equation: Ctrl-E
#       itemize: Ctrl-I
#     labels_anchors: false
#     latex_user_defs: false
#     report_style_numbering: false
#     user_envs_cfg: false
# ---

# %% [markdown]
# # Optimal Portfolio Choice over the Life Cycle
#
# Economists like to compare actual behavior to the choices that would be made by a "rational" agent who understands all the complexities of a decision, and who knows how to find the mathematically optimal choice taking those complexities into account.  "Behavioral" economics can be viewed as the attempt to understand how actual behavior deviates from an optimizing benchmark.
#
# The choice of your "risky portfolio share"
#
# .............
#
# .............

# %% [markdown]
# In this blog post we use the new class of Consumption Saving model `PortfolioConsumerType` to reproduce the paper, Consumption and Portfolio Choice Over the Life Cycle by Cocco, Gomes, & Maenhout (2005). (Explain the gist of paper?)
#
# We explain Portfolio Choice over various different assumptions and parameters.
#
#
# ![RShare_Means](figures/figure_Parameters_base/RShare_Means.png)

# %%
