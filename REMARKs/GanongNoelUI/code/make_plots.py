# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 21:44:30 2015

@author: ganong
"""


from rpy2 import robjects
from rpy2.robjects.lib import grid
from rpy2.robjects.packages import importr

# The R 'print' function
rprint = robjects.globalenv.get("print")
stats = importr('stats')
grdevices = importr('grDevices')
base = importr('base')
#datasets = importr('datasets')

grid.activate()

import rpy2.robjects.lib.ggplot2 as ggplot2
from rpy2.robjects.lib.ggplot2 import element_blank, \
                               theme_bw, \
                               theme, \
                               element_rect, \
                               element_text
                               
from rpy2.robjects import pandas2ri

import numpy as np
import pandas as pd
pandas2ri.activate()                                                        


robjects.r('''
    library(RColorBrewer)
    library(grid)
    #print(brewer.pal(8,"Set3")) Greys
    palette <- brewer.pal("Greys", n=9)
    #print(palette)
  color_background = "white"
  color_grid_major = palette[3]
  color_axis_text = palette[6]
  color_axis_title = palette[7]
  color_title = palette[9]
  #palette_lines <- brewer.pal("Dark2", n=3)
  palette_lines <- brewer.pal("Set2", n=8)
  palette_repeat <- c('#66C2A5', '#66C2A5', '#FC8D62','#FC8D62')
  linetype_repeat <- c("solid","dashed","solid","dashed")
''')

fte_theme = theme(**{'axis.ticks':element_blank(),
      'panel.background':element_rect(fill=robjects.r.color_background, color=robjects.r.color_background),
      'plot.background':element_rect(fill=robjects.r.color_background, color=robjects.r.color_background),
      'panel.border':element_rect(color=robjects.r.color_background), #'panel.grid.major':element_line(color=robjects.r.color_grid_major, size = 0.25),
      'panel.grid.minor':element_blank(),
      'axis.ticks':element_blank(),
      'legend.position':"right",
      'legend.background': element_rect(fill=robjects.r.color_background),
      'legend.text': element_text(size=10,color=robjects.r.color_axis_title),
      'legend.title': element_blank(),
      'plot.title':element_text(size=12, color=robjects.r.color_title,  vjust=1.25, hjust=0),
      'axis.text.x':element_text(size=10,color=robjects.r.color_axis_text),
      'axis.text.y':element_text(size=10,color=robjects.r.color_axis_text),
      'axis.title.x':element_text(size=10,color=robjects.r.color_axis_title, vjust=0),
      #'panel.grid.major':element_line(color=robjects.r.color_grid_major,size=.25),
      'axis.title.y':element_text(size=10,color=robjects.r.color_axis_title,angle=90)})


fte_theme_micro = theme(**{'axis.ticks':element_blank(),
      'panel.background':element_rect(fill=robjects.r.color_background, color=robjects.r.color_background),
      'plot.background':element_rect(fill=robjects.r.color_background, color=robjects.r.color_background),
      'panel.border':element_rect(color=robjects.r.color_background), #'panel.grid.major':element_line(color=robjects.r.color_grid_major, size = 0.25),
      'panel.grid.minor':element_blank(),
      'axis.ticks':element_blank(),
      'legend.position':"right",
      'legend.background': element_rect(fill=robjects.r.color_background),
      'legend.text': element_text(size=12,color=robjects.r.color_axis_title),
      'legend.title': element_blank(),
      'plot.title':element_text(size=14, color=robjects.r.color_title,  vjust=1.25, hjust=0),
      'axis.text.x':element_text(size=12,color=robjects.r.color_axis_text),
      'axis.text.y':element_text(size=12,color=robjects.r.color_axis_text),
      'axis.title.x':element_text(size=12,color=robjects.r.color_axis_title, vjust=0),
      #'panel.grid.major':element_line(color=robjects.r.color_grid_major,size=.25),
      'axis.title.y':element_text(size=12,color=robjects.r.color_axis_title,angle=90)})

pandas2ri.activate() 
#set up basic, repetitive plot features
base_plot = ggplot2.aes_string(x='mos_since_start', y='value',group='variable',colour='variable', shape = 'variable', linetype = 'variable')
line = ggplot2.geom_line()
point = ggplot2.geom_point() 
vert_line_onset = ggplot2.geom_vline(xintercept=-1.5, linetype=2, colour="#999999")           
vert_line_exhaust = ggplot2.geom_vline(xintercept=5.5, linetype=2, colour="#999999")   
vert_line_exhaust_FL = ggplot2.geom_vline(xintercept=3.5, linetype=2, colour="#999999")        
colors = ggplot2.scale_colour_manual(values=robjects.r.palette_lines)
hollow = ggplot2.scale_shape_manual(values=robjects.r('c(16,17,15,18,6,7,9,3)'))
xlab = ggplot2.labs(x="Months Since First UI Check")
loc_default = robjects.r('c(1,0)')
legend_f  = lambda loc = loc_default: ggplot2.theme(**{'legend.position':loc, 'legend.justification':loc})
ggsave = lambda filename, plot: robjects.r.ggsave(filename="../out/" + filename + ".pdf", plot=plot, width = 6, height = 4)

colors_alt = ggplot2.scale_colour_manual(values=robjects.r.palette_lines[1])
shape_alt = ggplot2.scale_shape_manual(values=17)



ggplot2_env = robjects.baseenv['as.environment']('package:ggplot2')

class GBaseObject(robjects.RObject):
    @classmethod
    def new(*args, **kwargs):
        args_list = list(args)
        cls = args_list.pop(0)
        res = cls(cls._constructor(*args_list, **kwargs))
        return res
        
class Annotate(GBaseObject):
    _constructor = ggplot2_env['annotate']
annotate = Annotate.new
